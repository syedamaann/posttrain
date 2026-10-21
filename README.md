# posttrain

Small-scale replica of [Rufus-Air](https://arxiv.org/abs/2609.29421): eight post-training stages on `Qwen/Qwen3-1.7B-Base`.

| Stage | Status | Weights | Data / logs |
|---|---|---|---|
| P0 setup | done | — | GPU + tokenizer checks |
| **P1 SFT (v4@300)** | **shipped** | [`syedamaan/qwen3-1.7b-rufus-s1`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1) — **not** ckpt-repo root (that is 422) | Data: [`syedamaan/rufus-s1-sft-data-v2`](https://huggingface.co/datasets/syedamaan/rufus-s1-sft-data-v2). W&B: [`s1-sft-v4-qwen3-1.7b`](https://forge.coreweave.com/wandb/i-dont-have-a-organisation/rufus-small/runs/t6a9bxe5). Eval: [`p1_sft/results_v4.jsonl`](p1_sft/results_v4.jsonl) |
| P2 reasoning RL | next | starts from S1@300 | vLLM + Math-Verify + GRPO/GSPO |
| P3–P8 | later | — | coding / IF / agents / RLHF |

**S1 pick: step 300.** Generation plateau; @400 closed/trunc got worse. Empty gens 0% (v3 bug gone). Thinking GSM8K `</think>` ~88%; no-think boxed ~98%. Acc ≈ base, not vendor.

## Load S1

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
REPO = "syedamaan/qwen3-1.7b-rufus-s1"
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype="auto")
tok = AutoTokenizer.from_pretrained(REPO)
```

## Layout

```
p0_setup/     env / GPU / tokenizer smoke test
p1_sft/       SFT train + frozen eval protocol
s1-model-card/  Hub README + ship script for @300
```

## Next

P2 from `syedamaan/qwen3-1.7b-rufus-s1`. Beat `@300` on the same 100 GSM8K + 100 MATH-500 protocol. Guardrail: no-think format.
