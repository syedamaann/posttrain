"""S2 reasoning RL: GSPO + closed boxed reward + DAPO + overlong penalty. KL=0."""
import json
import os
import sys
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import torch
import wandb
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from reward import closed_verify, shape_group, think_body

S1 = "syedamaan/qwen3-1.7b-rufus-s1"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
G = 8
MAX_NEW = 1024
OUT = "/workspace/s2-gspo-close"


def gsm8k_gold(ex):
    return ex["answer"].split("####")[-1].strip().replace(",", "")


def main():
    tok = AutoTokenizer.from_pretrained(S1)
    tok.eos_token = "<|im_end|>"
    tok.pad_token = "<|endoftext|>"

    raw = load_dataset("openai/gsm8k", "main", split="train")

    def row(ex):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"] + SUFFIX}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        return {"prompt": prompt, "answer": gsm8k_gold(ex)}

    ds = raw.map(row, remove_columns=raw.column_names)

    def verify_span(gold, span):
        return verify(parse(f"${gold}$"), parse(span))

    def math_r(completions, answer, **kwargs):
        binary, lengths = [], []
        for c, a in zip(completions, answer):
            text = c if isinstance(c, str) else c[-1]["content"]
            n_think = len(tok(think_body(text), add_special_tokens=False).input_ids)
            binary.append(closed_verify(text, a, verify_span, n_think))
            lengths.append(len(tok(text, add_special_tokens=False).input_ids))
        shaped = []
        for i in range(0, len(binary), G):
            shaped.extend(shape_group(binary[i : i + G], lengths[i : i + G], MAX_NEW))
        return shaped

    wandb.finish()
    args = GRPOConfig(
        output_dir=OUT, run_name="s2-gspo-close-qwen3-1.7b",
        max_steps=20, per_device_train_batch_size=G, num_generations=G,
        max_completion_length=MAX_NEW, learning_rate=1e-6,
        weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.98, max_grad_norm=1.0,
        bf16=True, gradient_checkpointing=True,
        importance_sampling_level="sequence", loss_type="grpo", beta=0.0,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
        temperature=1.0, top_p=0.95, top_k=20,
        logging_steps=1, save_strategy="steps", save_steps=10, save_only_model=True,
        report_to="wandb", remove_unused_columns=False, seed=0,
        mask_truncated_completions=True,
    )
    assert torch.cuda.memory_allocated() == 0, "GPU not empty: restart kernel"
    trainer = GRPOTrainer(
        model=S1, args=args, train_dataset=ds, reward_funcs=math_r, processing_class=tok,
    )
    trainer.train()
    tok.save_pretrained(OUT)
    for ckpt in Path(OUT).glob("checkpoint-*"):
        tok.save_pretrained(ckpt)
    log_path = Path(OUT) / "log_history.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(trainer.state.log_history, open(log_path, "w"))
    print("wrote", log_path)
    ckpt = f"{OUT}/checkpoint-20"
    print("eval: python /workspace/posttrain/s1/eval.py --model", ckpt, "--out /workspace/posttrain/s2/results-close.jsonl")


if __name__ == "__main__":
    main()
