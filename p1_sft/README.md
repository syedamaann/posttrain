# P1 — SFT (v4)

Capability floor: ChatML + two modes (`enable_thinking` True/False) + `\boxed{}`.

**Ship `checkpoint-300`.** Full ckpt dump: `syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts`. Data: `syedamaan/rufus-s1-sft-data-v2` (13,500 rows). W&B run `t6a9bxe5`.

## What ran

| version | change | result |
|---|---|---|
| v1 | empty think block in **completion** | model trained to emit empty `<think>` |
| v2 | mask fix; `group_by_length` | 50-step loss sawtooth; think-mode collapse at phase ckpts |
| v3 | `batch_rebalance` | smooth loss; think still broken (identical control embeddings) |
| **v4** | v3 + sub-token-mean init of 8 control rows | empty 0%; think tags work; **@300** plateau |

`train_v4.py` is the v4 recipe. `eval.py` is the frozen n=100 GSM8K + MATH-500 protocol. `results_v4.jsonl` is the A40 sweep (@100–@400; @422 not finished).

## Load

```python
AutoModelForCausalLM.from_pretrained(
    "syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts", subfolder="checkpoint-300")
```
