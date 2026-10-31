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

A proposed map, not a settled plan. S1–S3 describe what this repo has used. S4–S8 are suggestions. The specific tools get checked again at the start of each stage.

Two rules. One new major tool per stage, so a break has a suspect. Stay on TRL until a stage needs something TRL does not fit. A later stage rescores the frozen math eval above.

| Stage | What the paper trains | Small-scale version | New tool to learn | Status |
|---|---|---|---|---|
| S1 SFT | 9.01M samples, six categories | `rufus-s1-sft-data-v2`, checkpoint-300 | TRL SFT, chat templates, loss masking, Hub model cards | done |
| S2 Reasoning RL | GSPO, Math-Verify, length penalty | GSM8K, 1,400 steps, groups kept in (0, 0.8] | TRL GRPO/GSPO, vLLM colocate, Math-Verify, W&B on CoreWeave Forge, the frozen eval | done |
| S3 Coding RL (§3.3) | execute the tests, binary reward, about 29k problems | 200 EvolveCoder assert problems | sandboxed execution, Modal (Sandbox, Function, Secret, Image), Hub branches, paired comparison | underway, no training steps |
| S4 Instruction-following RL (§3.4) | rubric reward, Python checks plus an LLM judge, every check must pass | Python-checkable constraints first, judge rubrics second | vLLM as a separate OpenAI-compatible server (TRL server mode), LLM-as-judge, lm-evaluation-harness for IFEval/IFBench | not started |
| S5 General Agent (§3.5) | multi-turn MCP tool use (AgentWorldModel), binary check of environment state | a few small MCP tool environments | framework switch to prime-rl + verifiers, or SkyRL; MCP; multi-turn rollouts; token-in/token-out; tool-call parsing; publish an environment | not started |
| S6 Coding Agent (§3.6) | terminal tasks in Firecracker VMs, with aborted runs tagged | small terminal tasks, not SWE-bench | Harbor task format, a sandbox that keeps state across turns (Modal or E2B), images built ahead of time, infra-failure tagging | not started |
| S7 Search Agent (§3.7) | web search and page extraction, LLM-judge reward, Dr. GRPO, a small KL anchor | local retrieval over a bounded corpus, not a paid search API | retrieval (BM25 or a dense index), judge-reward noise, mean-only advantages, SGLang for multi-turn caching | not started |
| S8 RLHF | on-policy GRPO against an open reward model, length penalty | a small open reward model | reward models, overoptimization and reward hacking, DPO as a comparison, a judged eval | not started |
| Capstone | | one notebook, `RETRAIN = False`, loads every published checkpoint | that notebook, a Hub Collection, the essay, the video | not started |

The S3 Function is a tool to learn. Whether eval runs actually move onto a Modal Function is still open. prime-rl, SkyRL, Harbor, E2B, and SGLang are the candidates named for later stages, not choices already made.

Inference, one topic per stage:

| Stage | Inference topic |
|---|---|
| S1 | `generate()`, sampling parameters, special tokens |
| S2 | vLLM batching, the colocate memory split, throughput |
| S3 | generation cost against execution cost |
| S4 | serving with vLLM, prefix caching, request batching |
| S5 | chat-template fidelity, retokenization drift |
| S6 | long contexts, KV-cache budgets |
| S7 | SGLang's prefix cache for multi-turn reuse, against vLLM |
| S8 and the capstone | quantization, speculative decoding, a latency and throughput measurement of the shipped model |

S5–S7 are where shrinking the paper gets hard. A rollout is many tool calls, and a 1.7B model may barely move the paper's agent exams (tau2-bench, Terminal-Bench, BrowseComp). Each of those stages needs a smaller exam it can actually change, chosen the way S3's holdout is chosen, and its own budget. The paper budgets the agent stages separately rather than by an average. When a stage starts, recheck the small reward models available then, whether prime-rl or SkyRL is comfortable on one GPU, and the current Harbor and Modal APIs.

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
