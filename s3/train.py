"""S3 coding RL pilot: GSPO on the 200 easiest mixed EvolveCoder problems.

The corpus was graded at temperature 0.6, so training samples at 0.6 too.
Stage 2 trained at 1.0. An optimizer step averages ACCUM prompt groups, so
one group of eight is not a whole update. One pass over the 200 problems
is 200 / ACCUM optimizer steps. MAX_STEPS is four passes.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import score_completion, shape_group

G = 8
ACCUM = 8
MAX_NEW = 2048
MAX_STEPS = 100
START = "syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl"
DATA = "syedamaan/rufus-s3-coding-pilot"
OUT = "/workspace/s3-coding-pilot"


def system_text() -> str:
    marker = "`" * 3
    return (
        "Reason about the problem, then write </think>. "
        "After that, put the final solution in one Python block that opens with "
        + marker + "python."
    )


def chat_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": system_text()},
        {"role": "user", "content": question},
    ]


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def code_rewards(completions, problem_ids, tests, run, group_size=G):
    """Score a flat batch of completions and shape each group of `group_size`.

    TRL repeats each prompt's columns once per generation, so the three
    lists are the same length and already laid out in groups.
    """
    if not (len(completions) == len(problem_ids) == len(tests)):
        raise RuntimeError(
            f"reward columns misaligned: {len(completions)} completions, "
            f"{len(problem_ids)} ids, {len(tests)} test lists"
        )
    if len(completions) % group_size != 0:
        raise RuntimeError(
            f"{len(completions)} completions is not a multiple of group size {group_size}"
        )
    binary = []
    for completion, problem_id, case in zip(completions, problem_ids, tests):
        scored = score_completion(completion_text(completion), problem_id, list(case), run)
        binary.append(scored["reward"])
    shaped = []
    for i in range(0, len(binary), group_size):
        shaped.extend(shape_group(binary[i : i + group_size]))
    return shaped


def main():
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(Path(__file__).parent))

    import torch
    import wandb
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from sandbox import run_tests

    tok = AutoTokenizer.from_pretrained(START)
    tok.eos_token = "<|im_end|>"
    tok.pad_token = "<|endoftext|>"

    raw = load_dataset(DATA, split="train")

    def row(ex):
        prompt = tok.apply_chat_template(
            chat_messages(ex["question"]),
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        return {"prompt": prompt, "problem_id": ex["id"], "tests": ex["tests"]}

    ds = raw.map(row, remove_columns=raw.column_names)

    def code_r(completions, problem_id, tests, **kwargs):
        return code_rewards(completions, problem_id, tests, run_tests, G)

    wandb.finish()
    args = GRPOConfig(
        output_dir=OUT, run_name="s3-coding-pilot",
        max_steps=MAX_STEPS, per_device_train_batch_size=G, num_generations=G,
        gradient_accumulation_steps=ACCUM,
        max_completion_length=MAX_NEW, learning_rate=1e-6,
        weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.98, max_grad_norm=1.0,
        bf16=True, gradient_checkpointing=True,
        importance_sampling_level="sequence", loss_type="grpo", beta=0.0,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
        temperature=0.6, top_p=0.95, top_k=20,
        logging_steps=1, save_strategy="steps", save_steps=25, save_total_limit=4,
        save_only_model=True,
        report_to="wandb", remove_unused_columns=False, seed=0,
        mask_truncated_completions=True,
    )
    assert torch.cuda.memory_allocated() == 0, "GPU not empty: restart kernel"
    trainer = GRPOTrainer(
        model=START, args=args, train_dataset=ds, reward_funcs=code_r, processing_class=tok,
    )
    trainer.train()
    tok.save_pretrained(OUT)
    for ckpt in Path(OUT).glob("checkpoint-*"):
        tok.save_pretrained(ckpt)
    log_path = Path(OUT) / "log_history.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(trainer.state.log_history, open(log_path, "w"))
    print("wrote", log_path)


if __name__ == "__main__":
    main()
