# posttrain

Small-scale [Rufus-Air](https://arxiv.org/abs/2609.29421) replica on `Qwen/Qwen3-1.7B-Base`.

| Stage | Status | Artifact |
|---|---|---|
| **S1 SFT** | done | [`syedamaan/qwen3-1.7b-rufus-s1`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1) is **checkpoint-300**. Do not load [`…-sft-v4-ckpts`](https://huggingface.co/syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts) at repo root (that is 422). |
| S2 reasoning RL | next | beat S1 with `s1/eval.py` |

```
s1/train.py          SFT recipe
s1/eval.py           frozen n=100 GSM8K + MATH-500
s1/results.jsonl     A40 sweep (base … @400)
s1/make_figures.py   regenerates card PNGs
s1/card/             Hub README + figures + ship.py
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "syedamaan/qwen3-1.7b-rufus-s1"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
```

P2 loads that repo. Beat GSM8K think **78%** and MATH-500 think trunc **22%** without breaking no-think boxed **98%** / closed **88%**.
