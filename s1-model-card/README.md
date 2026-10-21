---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B-Base
language:
  - en
library_name: transformers
tags:
  - qwen3
  - sft
  - post-training
  - chain-of-thought
  - rufus-air
pipeline_tag: text-generation
---

# qwen3-1.7b-rufus-s1

SFT stage (S1) of a small-scale replica of [*Rufus-Air: An Open LLM Post-Training Recipe*](https://arxiv.org/abs/2609.29421).

**This repo is `checkpoint-300` only.** Do not use `syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts` at repo root — that `model.safetensors` is **step 422**, past the eval plateau.

| | |
|---|---|
| Student | `Qwen/Qwen3-1.7B-Base` (1.7B, tied embeddings) |
| Recipe | 1 epoch SFT, 13.5k prompt-completion rows, AdamW 2e-5 cosine, batch 32 |
| Pick | step **300** / 422 (generation plateau; 400+ got worse) |
| Data | `syedamaan/rufus-s1-sft-data-v2` (private) |

## What S1 is for

Not “beat Qwen’s official instruct model.” S1 installs a **capability floor**: ChatML turns, two generation modes, `\boxed{}` answers, stop on `<|im_end|>`. Later RL (P2+) should beat **this** checkpoint, not vendor.

## Before vs after (same weights family)

`enable_thinking` is a **prompt** flag on one checkpoint. Base never emits think tags. After S1, thinking mode writes a CoT block; non-thinking does not.

```
BEFORE (Qwen3-1.7B-Base)                 AFTER (this repo)
enable_thinking=True                     enable_thinking=True
assistant                                assistant
<short direct answer>                    <think>
                                         Okay, … (long trace)
                                         </think>
                                         The answer is \boxed{18}.

enable_thinking=False                    enable_thinking=False
(empty think block already in prompt)    (empty think block already in prompt)
<answer, sometimes messy boxing>         <short answer, \boxed{}, no extra think tags>
```

Quantitative (n=100, Math-Verify, temp 0.6 / top_p 0.95 / top_k 20; **this protocol, A40**):

| model | set | mode | acc | boxed | `</think>` | empty | trunc | tok |
|---|---|---|---|---|---|---|---|---|
| base | gsm8k | think | 82% | 88% | 0% | 0% | 0% | 274 |
| **s1@300** | gsm8k | think | **78%** | 89% | **88%** | **0%** | 11% | 1319 |
| base | gsm8k | no-think | 65% | 70% | 0% | 0% | 14% | 361 |
| **s1@300** | gsm8k | no-think | **70%** | **98%** | **0%** | 0% | 1% | 184 |
| base | math500 | no-think | 38% | 55% | 0% | 0% | 37% | 624 |
| **s1@300** | math500 | no-think | **58%** | **89%** | 0% | 0% | 9% | 374 |

Thinking-mode GSM8K **accuracy** did not leap past base. What changed is **protocol**: empty gens 0%, `</think>` closed ~88%, no-think boxed 98% with no tag leak. MATH-500 think is still ~60% and ~22% truncated at 4096 (length, not control-token collapse).

v1–v3 taught three bugs (mask, length-grouped curriculum, **identical embeddings** for `<think>` / `</think>` / `<|im_end|>`). v4 = v3 + sub-token-mean init of those eight rows. Ship v4@300.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "syedamaan/qwen3-1.7b-rufus-s1"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")

def generate(question, think=True, max_new=2048):
    messages = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new,
        do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        eos_token_id=[151645, 151643],  # <|im_end|>, <|endoftext|>
        pad_token_id=151643,
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

print(generate("What is 17*23?", think=True))
print(generate("What is 17*23?", think=False))
```

- `enable_thinking=True` → prompt ends at `assistant\n`; model should open `<think>`.
- `enable_thinking=False` → template inserts empty `<think>\n\n</think>\n\n`; model should answer only.

## Training (v4)

- Data: 13,500 rows (OpenR1 math-think, Tulu chat/code/IF/GSM8K-short, Hermes tools). Token mix is ~75% math-think.
- `SFTTrainer`, `completion_only_loss`, `batch_rebalance`, no packing.
- Before train: eight control-token embedding rows were the **same placeholder**; replaced with mean of spelling sub-tokens (Hewitt-style). Tied input/output embeddings.
- LR 2e-5, warmup 5%, cosine, AdamW β₂=0.95, wd 0.1, clip 1.0, bf16, grad checkpointing. 422 steps × batch 32. W&B: `s1-sft-v4-qwen3-1.7b`.

## What this is not

- Not `Qwen/Qwen3-1.7B` (vendor instruct / distillation ceiling, GSM8K think 86% under the same protocol).
- Not RL. P2 should start **here** and beat these numbers on math without breaking no-think format.
- Eval n=100 is noisy (~5 pp). Compare models on this table, not to other protocols.

## Checkpoints

All steps: [`syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts) (`checkpoint-{100,200,300,400,422}`). Prefer **300**.
