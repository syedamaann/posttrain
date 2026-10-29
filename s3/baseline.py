"""Baseline the S2 checkpoint on a frozen EvolveCoder holdout.

Run this on the GPU machine, not the laptop. It downloads the dataset there.
The first run records format only. Pass --score to grade extracted code in Modal.

The holdout ids are written next to the metrics. Training must exclude them.
Sampling matches s1/eval.py: temperature 0.6, top_p 0.95, top_k 20.
The math eval's "put the answer in \\boxed{}" suffix is intentionally absent.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from filters import keep_row
from reward import format_diagnostics, score_completion


def usable(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept = []
    for row in rows:
        ok, _ = keep_row(row, seen, exam=[])
        if ok:
            kept.append(row)
    return kept


def choose_holdout(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    ordered = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(ordered)
    if n > len(ordered):
        raise ValueError(f"asked for {n} holdout problems, have {len(ordered)}")
    return ordered[:n]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "fenced": sum(r["fenced_python"] for r in records) / n,
        "boxed_leak": sum(r["boxed_leak"] for r in records) / n,
        "trunc": sum(r["trunc"] for r in records) / n,
        "tokens": sum(r["tokens"] for r in records) / n,
        "prompt_skip": sum(r["prompt_skip"] for r in records),
    }
    scored = [r for r in records if "reward" in r and not r["prompt_skip"]]
    if scored:
        out["scored"] = len(scored)
        out["pass"] = sum(r["reward"] == 1.0 for r in scored) / len(scored)
        out["abort"] = sum(r["reward"] is None for r in scored) / len(scored)
    return out


def generate(model, tok, questions, max_new, max_prompt, batch):
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    tok.padding_side = "left"
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for q in questions
    ]
    encoded = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    texts = [""] * len(prompts)
    lengths = [0] * len(prompts)
    skipped = [False] * len(prompts)
    import torch
    for start in range(0, len(prompts), batch):
        chunk = list(range(start, min(start + batch, len(prompts))))
        fit = [i for i in chunk if len(encoded[i]) <= max_prompt]
        for i in chunk:
            if i not in fit:
                skipped[i] = True
        if not fit:
            continue
        b = tok([prompts[i] for i in fit], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **b, max_new_tokens=max_new, do_sample=True,
            temperature=0.6, top_p=0.95, top_k=20,
            pad_token_id=tok.pad_token_id, eos_token_id=eos,
        )
        for row, i in enumerate(fit):
            gen = out[row, b.input_ids.shape[1] :]
            n = int((gen != tok.pad_token_id).sum())
            texts[i] = tok.decode(gen, skip_special_tokens=True)
            lengths[i] = n
    return texts, lengths, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--max-prompt", type=int, default=4096)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--score", action="store_true")
    p.add_argument("--out", default="s3/baseline.jsonl")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = load_dataset("TIGER-Lab/EvolveCoder-22K", split="train")
    rows = usable([{"id": r["id"], "question": r["question"], "tests": r["tests"]} for r in raw])
    holdout = choose_holdout(rows, args.n, args.seed)
    ids_path = Path(args.out).with_suffix(".ids.json")
    ids_path.write_text(json.dumps([r["id"] for r in holdout]))
    print("holdout", len(holdout), "ids", ids_path, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.eos_token = "<|im_end|>"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    texts, lengths, skipped = generate(
        model, tok, [r["question"] for r in holdout], args.max_new, args.max_prompt, args.batch,
    )
    run = None
    if args.score:
        from modal_sandbox import run_tests
        run = run_tests

    records = []
    for row, text, n, skip in zip(holdout, texts, lengths, skipped):
        diag = format_diagnostics(text)
        rec = {
            "id": row["id"],
            "fenced_python": diag["fenced_python"],
            "boxed_leak": diag["boxed_leak"],
            "trunc": (not skip) and n >= args.max_new,
            "tokens": n,
            "prompt_skip": skip,
        }
        if run is not None and not skip:
            scored = score_completion(text, row["id"], row["tests"], run)
            rec["reward"] = scored["reward"]
            rec["reason"] = scored["reason"]
        records.append(rec)
        print(rec["id"], "fence", rec["fenced_python"], "box", rec["boxed_leak"], "trunc", rec["trunc"], flush=True)

    summary = {"model": args.model, "set": "evolvecoder-holdout", "k": 1, **summarize(records)}
    print(summary, flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps({"summary": summary, "records": records}) + "\n")


if __name__ == "__main__":
    main()
