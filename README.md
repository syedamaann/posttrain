# posttrain

Small-scale replica of [Rufus-Air](https://arxiv.org/abs/2609.29421): eight post-training stages on `Qwen/Qwen3-1.7B-Base`.

| Stage | Status | Weights | Data / logs |
|---|---|---|---|
| P0 setup | done | — | GPU + tokenizer checks |
| **P1 SFT (v4@300)** | **ship this** | Hub ckpts: [`syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts) **`subfolder=checkpoint-300`** — **not repo root (that is 422)** | Data: [`syedamaan/rufus-s1-sft-data-v2`](https://huggingface.co/datasets/syedamaan/rufus-s1-sft-data-v2). W&B: [`s1-sft-v4-qwen3-1.7b`](https://forge.coreweave.com/wandb/i-dont-have-a-organisation/rufus-small/runs/t6a9bxe5). Eval: [`p1_sft/results_v4.jsonl`](p1_sft/results_v4.jsonl) |
| P2 reasoning RL | next | starts from S1@300 | vLLM + Math-Verify + GRPO/GSPO |
| P3–P8 | later | — | coding / IF / agents / RLHF |

**S1 pick: step 300.** Generation plateau; @400 closed/trunc got worse. Empty gens 0% (v3 bug gone). Thinking GSM8K `</think>` ~88%; no-think boxed ~98%. Acc ≈ base, not vendor.

## Load S1

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
REPO = "syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts"
model = AutoModelForCausalLM.from_pretrained(REPO, subfolder="checkpoint-300", dtype="auto")
tok = AutoTokenizer.from_pretrained(REPO, subfolder="checkpoint-300")
```

After the next GPU session this should become `syedamaan/qwen3-1.7b-rufus-s1` (clean copy + model card in `s1-model-card/`).

## Layout

```
p0_setup/     env / GPU / tokenizer smoke test
p1_sft/       SFT train + frozen eval protocol
s1-model-card/  Hub README + ship script for @300
```

## Next

1. GPU in the **network-volume region** (CA-MTL-1): copy `Untitled.ipynb` off the volume; upload `@300` to a clean Hub repo.
2. P2 from that repo. Beat `@300` on the same 100 GSM8K + 100 MATH-500 protocol. Guardrail: no-think format.
