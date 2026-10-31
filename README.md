# posttrain

Small-scale [Rufus-Air](https://arxiv.org/abs/2609.29421) replica on `Qwen/Qwen3-1.7B-Base`. The paper is in [`papers/rufus-air.pdf`](papers/rufus-air.pdf). Each stage loads the previous stage's Hub checkpoint.

| Stage | Status | Artifact |
|---|---|---|
| **S1 SFT** (§3.1) | done | [`syedamaan/qwen3-1.7b-rufus-s1`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1) is **checkpoint-300**. Do not load [`…-sft-v4-ckpts`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts) at repo root (that is 422). |
| **S2 reasoning RL** (§3.2) | done | [`syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl) (private). Card, figures, and the 1400-step path are in [`s2/card/`](s2/card/README.md). |
| S3 coding RL (§3.3) | underway | Code is in [`s3/`](s3/). No training steps yet. Load the S2 repo. The reward is an assert sandbox. |

Eval for S1 and S2 is frozen `s1/eval.py`: n=100 GSM8K + MATH-500, temp 0.6 / top_p 0.95 / top_k 20, think at 4096, no-think at 1024. n=100 is about 5 points of noise.

S2 against S1: GSM8K think accuracy **78% → 83%**, MATH-500 think truncation **22% → 11%**, GSM8K think close **88% → 99%**. No-think boxed stayed at 99% and 90%. The extra no-think `</think>` is **0%** on both sets. A later stage has to keep that table.

## How the rest proceeds

A proposed map, not a settled plan. S1–S3 are what this repo has done. Later rows are the paper's next stages and a guess at the one new tool worth learning there. The guess is checked at the start of that stage.

Two rules. One new major tool per stage, so a break has a suspect. Stay on TRL until a stage needs something TRL does not fit. Each stage also answers one inference question. A later stage rescores the frozen math eval above.

| Stage | What the paper trains | Small-scale version | Status |
|---|---|---|---|
| S1 SFT | 9.01M samples, six categories | `rufus-s1-sft-data-v2`, checkpoint-300 | done |
| S2 Reasoning RL | verifiable math reward | GSM8K, 1,400 GSPO steps | done |
| S3 Coding RL (§3.3) | execute the program | 200 assert problems, sandbox reward | underway, no training steps |
| S4 Instruction-following RL (§3.4) | a rubric the whole response must pass | constraints a Python check can score, before any judge | not started |
| S5 General Agent (§3.5) | MCP tools, then a check of the environment's state | a few small tool environments | not started |
| S6 Coding Agent (§3.6) | multi-turn shell tasks in a VM that keeps its files and processes | small terminal tasks | not started |
| S7 Search Agent (§3.7) | search, then an LLM judge | a bounded corpus before the open web | not started |
| S8 RLHF | a learned reward on open-ended answers | a small open reward model, beside a non-RL baseline | not started |

S3's new tool is sandboxed execution, Modal included. Intermediate checkpoints go to Hub branches so eval can leave the training machine. A separate serving process, a second training framework, a stateful VM, a retriever, and a reward model belong to S4–S8. Those libraries stay unnamed until the stage starts.

Inference, one question per stage: sampling and special tokens, batched generation and the trainer/generator memory split, whether generation or execution is the slow part, then serving, chat-template fidelity, the cost of a long context, reuse of a multi-turn cache, and a latency and throughput measurement of the shipped model.

S5–S7 are where a 1.7B copy gets hard. A rollout is many tool calls, and the paper's agent exams may not move. Each of those stages needs a smaller exam it can actually change, and its own budget. The finish is one notebook that loads the published checkpoints, a Hub collection, and a written account of the choices.

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
