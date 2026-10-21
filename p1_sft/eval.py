"""Frozen S1 eval: 100 GSM8K + 100 MATH-500, think@4096 and no-think@1024."""
import argparse
import json
import os

import torch
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def score(text, gold):
    try:
        return bool(verify(parse(f"${gold}$"), parse(text.split("</think>")[-1])))
    except Exception:
        return False


@torch.no_grad()
def eval_set(model, tok, items, enable_thinking, max_new_tokens, bs=25, seed=0):
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    torch.manual_seed(seed)
    tok.padding_side = "left"
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q + SUFFIX}],
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        for q, _ in items
    ]
    acc = boxed = trunc = closed = empty = 0
    lens = []
    for i in range(0, len(prompts), bs):
        print("batch", i, flush=True)
        b = tok(prompts[i : i + bs], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **b, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.6, top_p=0.95, top_k=20,
            pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
        )
        for j, g in enumerate(out[:, b.input_ids.shape[1] :]):
            t = tok.decode(g, skip_special_tokens=True)
            n = int((g != tok.pad_token_id).sum())
            lens.append(n)
            empty += t.strip() == ""
            trunc += n >= max_new_tokens
            closed += "</think>" in t
            boxed += "\\boxed" in t.split("</think>")[-1]
            acc += score(t, items[i + j][1])
    n = len(items)
    return dict(
        acc=acc / n, boxed=boxed / n, closed=closed / n, empty=empty / n,
        trunc=trunc / n, tokens=sum(lens) / n,
    )


def load_sets():
    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(100))
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(100))
    return {
        "gsm8k": [(ex["question"], ex["answer"].split("####")[-1].strip().replace(",", "")) for ex in gsm],
        "math500": [(ex["problem"], ex["answer"]) for ex in m500],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--subfolder", default=None)
    p.add_argument("--out", default="p1_sft/results_v4.jsonl")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, subfolder=args.subfolder)
    tok.eos_token = "<|im_end|>"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, subfolder=args.subfolder, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    sets = load_sets()
    for set_name, items in sets.items():
        for mode, think, budget in [("think", True, 4096), ("no-think", False, 1024)]:
            r = eval_set(model, tok, items, think, budget)
            row = {"model": args.subfolder or args.model, "set": set_name, "mode": mode, **r}
            print(row)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
