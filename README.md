# posttrain

Small-scale [Rufus-Air](https://arxiv.org/abs/2609.29421) replica on `Qwen/Qwen3-1.7B-Base`. The paper is in [`papers/rufus-air.pdf`](papers/rufus-air.pdf). Each stage loads the previous stage's Hub checkpoint.

| Stage | Status | Artifact |
|---|---|---|
| **S1 SFT** (§3.1) | done | [`syedamaan/qwen3-1.7b-rufus-s1`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1) is **checkpoint-300**. Do not load [`…-sft-v4-ckpts`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts) at repo root (that is 422). |
| **S2 reasoning RL** (§3.2) | done | [`syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl) (private). Card, figures, and the 1400-step path are in [`s2/card/`](s2/card/README.md). |
| S3 coding RL (§3.3) | not started | Load the S2 repo. A code sandbox is the reward. |

Eval for S1 and S2 is frozen `s1/eval.py`: n=100 GSM8K + MATH-500, temp 0.6 / top_p 0.95 / top_k 20, think at 4096, no-think at 1024. n=100 is about 5 points of noise.

S2 against S1: GSM8K think accuracy **78% → 83%**, MATH-500 think truncation **22% → 11%**, GSM8K think close **88% → 99%**. No-think boxed stayed at 99% and 90%. The extra no-think `</think>` is **0%** on both sets. A later stage has to keep that table.

```
papers/rufus-air.pdf paper
s1/train.py          SFT recipe
s1/eval.py           frozen math eval
s1/results.jsonl     S1 sweep, base through step 400
s1/make_figures.py   S1 card figures
s1/card/             S1 Hub card
s2/reward.py         closed think, boxed Math-Verify, DAPO (0, 0.8], overlong decay
s2/test_reward.py    reward locks, including the no-think duplicate
s2/train.py          GSPO, KL=0, vLLM colocate, GSM8K train
s2/make_figures.py   S2 card figures
s2/card/             S2 Hub card
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
```

`enable_thinking=True` asks the model to open `<think>`, close `</think>`, and box the answer. `enable_thinking=False` already closes an empty think in the prompt, and the completion should not add another tag.
