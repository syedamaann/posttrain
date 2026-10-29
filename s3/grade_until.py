"""Grade queued EvolveCoder problems until the keep count hits the target.

The notebook writes /tmp/s3_until200.json, then runs this file with the
vLLM interpreter on the pod. A keep is a group whose shaped rewards still
contain a 1: at least one full pass, and a mean of at most 0.8.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from reward import score_completion, shape_group
from sandbox import run_tests

blob = json.load(open("/tmp/s3_until200.json"))
rows, target, keeps = blob["rows"], blob["target"], blob["keeps_now"]
model_id = "syedamaan/qwen3-1.7b-rufus-s2-reasoning-rl"
tok = AutoTokenizer.from_pretrained(model_id)
marker = "`" * 3
system = (
    "Reason about the problem, then write </think>. "
    "After that, put the final solution in one Python block that opens with "
    + marker + "python."
)
print("loading", flush=True)
llm = LLM(
    model=model_id,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=32,
    gpu_memory_utilization=0.90,
)
params = SamplingParams(n=8, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048)
done = json.load(open("/workspace/s3_solvability.json"))
graded = 0
for start in range(0, len(rows), 4):
    if keeps >= target:
        break
    chunk = rows[start:start + 4]
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for row in chunk
    ]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    for row, out in zip(chunk, outputs):
        rewards = [
            score_completion(comp.text, row["id"], row["tests"], run_tests)["reward"]
            for comp in out.outputs
        ]
        shaped = shape_group(rewards)
        status = "keep" if any(x == 1.0 for x in shaped) else "drop"
        done[row["id"]] = {"rewards": rewards, "shaped": shaped, "status": status}
        with open("/workspace/s3_solvability.json", "w") as f:
            json.dump(done, f)
            f.flush()
        graded += 1
        keeps += status == "keep"
        print(status, row["id"], rewards, "keeps", keeps, "graded", graded, flush=True)
        if keeps >= target:
            break
print("finished", "keeps", keeps, "graded", graded, flush=True)
