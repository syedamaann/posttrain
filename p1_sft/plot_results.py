"""CPU plot of p1_sft/results_v4.jsonl → p1_sft/v4_eval.png"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).parent
rows = [json.loads(l) for l in (here / "results_v4.jsonl").read_text().splitlines() if l.strip()]
df = pd.DataFrame(rows)
df["step"] = df["model"].map(lambda m: 0 if m == "base" else int(str(m).split("@")[-1]))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, metric in zip(axes.flat, ["acc", "closed", "trunc", "tokens"]):
    for (s, mode), g in df.groupby(["set", "mode"]):
        g = g.sort_values("step")
        ax.plot(g["step"], g[metric], marker="o", label=f"{s} {mode}")
    ax.set_title(metric)
    ax.set_xlabel("SFT step (0 = base)")
    if metric != "tokens":
        ax.set_ylim(-0.05, 1.05)
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = here / "v4_eval.png"
fig.savefig(out, dpi=120)
print("wrote", out)
