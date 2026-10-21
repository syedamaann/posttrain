#!/usr/bin/env python3
"""Write s1/card/{protocol,metrics,sweep}.png from s1/results.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
CARD = HERE / "card"
ROWS = [
    json.loads(l)
    for l in (HERE / "results.jsonl").read_text().splitlines()
    if l.strip()
]

INK = "#18181b"
MUTED = "#71717a"
RULE = "#e4e4e7"
PANEL = "#fafafa"
S1_C = "#1d4ed8"
S1_FILL = "#eff6ff"
CHIP = "#ffffff"
CHIP_EDGE = "#d4d4d8"
BG = "#ffffff"
BASE_BAR = "#a1a1aa"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.linewidth": 0.8,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    }
)


def _row(model: str, set_: str, mode: str) -> dict:
    hits = [r for r in ROWS if r["model"] == model and r["set"] == set_ and r["mode"] == mode]
    if not hits:
        raise KeyError((model, set_, mode))
    return hits[0]


def _step(model: str) -> int:
    return 0 if model == "base" else int(str(model).split("@")[-1])


def _chip(ax, x, y, text, *, accent=False, fontsize=8):
    fc, ec, tc = (S1_FILL, S1_C, S1_C) if accent else (CHIP, CHIP_EDGE, INK)
    w = 0.026 + len(text) * 0.0118 * (fontsize / 8)
    h = 0.072
    ax.add_patch(
        FancyBboxPatch(
            (x, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.9,
            facecolor=fc,
            edgecolor=ec,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fontsize, color=tc, fontfamily="DejaVu Sans Mono", zorder=3)
    return x + w


def _flow(ax, items, x, y):
    for i, (text, accent) in enumerate(items):
        x = _chip(ax, x, y, text, accent=accent)
        if i < len(items) - 1:
            ax.text(x + 0.018, y, "→", ha="center", va="center", fontsize=10, color=MUTED, zorder=3)
            x += 0.036
    return x


def protocol():
    fig = plt.figure(figsize=(11.4, 6.15))
    fig.text(0.5, 0.97, "Same answer. Different protocol.", ha="center", va="top", fontsize=15)
    fig.text(
        0.5,
        0.925,
        "Natalia sold 48 clips in April, half as many in May.  Thinking on.  Greedy.  Both models output 72.",
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )

    ax_l = fig.add_axes([0.035, 0.08, 0.45, 0.78])
    ax_r = fig.add_axes([0.515, 0.08, 0.45, 0.78])
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.0),
                1.0,
                1.0,
                boxstyle="round,pad=0.012,rounding_size=0.028",
                linewidth=0.9,
                facecolor=PANEL,
                edgecolor=RULE,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    ax_l.text(0.5, 0.93, "Base  ·  Qwen3-1.7B", ha="center", va="center", fontsize=10, color=MUTED)
    ax_r.text(0.5, 0.93, "S1  ·  this repo  @300", ha="center", va="center", fontsize=10, color=S1_C)

    _flow(ax_l, [("72", False), ("<|endoftext|>", False)], 0.07, 0.80)
    _flow(ax_r, [("<think>", True), ("CoT", False), ("</think>", True)], 0.07, 0.82)
    _flow(ax_r, [(r"\boxed{72}", True), ("<|im_end|>", True)], 0.07, 0.72)

    ax_l.text(0.07, 0.68, "No think tags. Stops on the base EOS.", fontsize=8, color=MUTED)
    ax_r.text(0.07, 0.62, "Opens think, closes it, boxes, ChatML EOS.", fontsize=8, color=MUTED)

    ax_l.text(
        0.07,
        0.58,
        "Natalia sold 48 clips in April and half as many\n"
        "in May, which is 48/2 = 24 clips. So, Natalia\n"
        "sold 48 + 24 = 72 clips altogether in April\n"
        "and May.<|endoftext|>",
        ha="left",
        va="top",
        fontsize=8,
        fontfamily="DejaVu Sans Mono",
        linespacing=1.5,
        color=INK,
    )
    ax_r.text(
        0.07,
        0.54,
        "<think>\n"
        "Okay, let's see. Natalia sold clips … April: 48.\n"
        "May: 24. … the answer should be 72.\n"
        r"The total is \boxed{72}."
        "\n"
        "</think>\n"
        "\n"
        "Thus, the total number of clips Natalia sold\n"
        r"in April and May is \boxed{72}.<|im_end|>",
        ha="left",
        va="top",
        fontsize=8,
        fontfamily="DejaVu Sans Mono",
        linespacing=1.5,
        color=INK,
    )
    fig.text(
        0.5,
        0.015,
        "S1 is a capability floor (ChatML, two modes, boxed answers). Later RL should beat this checkpoint, not vendor instruct.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    fig.savefig(CARD / "protocol.png", dpi=180)
    plt.close(fig)


def metrics():
    pairs = [
        ("GSM8K think\n</think> closed", 100 * _row("base", "gsm8k", "think")["closed"], 100 * _row("v4@300", "gsm8k", "think")["closed"]),
        ("GSM8K no-think\nboxed", 100 * _row("base", "gsm8k", "no-think")["boxed"], 100 * _row("v4@300", "gsm8k", "no-think")["boxed"]),
        ("MATH-500 no-think\nboxed", 100 * _row("base", "math500", "no-think")["boxed"], 100 * _row("v4@300", "math500", "no-think")["boxed"]),
        ("MATH-500 no-think\ntruncated  ↓", 100 * _row("base", "math500", "no-think")["trunc"], 100 * _row("v4@300", "math500", "no-think")["trunc"]),
        ("GSM8K think\naccuracy", 100 * _row("base", "gsm8k", "think")["acc"], 100 * _row("v4@300", "gsm8k", "think")["acc"]),
    ]
    cards = [
        ("Closed think", "0%  →  88%", "GSM8K  </think>"),
        ("No-think boxed", "70%  →  98%", "GSM8K"),
        ("Think accuracy", "82%  →  78%", "not the S1 win"),
        ("Think tokens", "274  →  1319", "5× longer CoT"),
    ]

    fig = plt.figure(figsize=(11.4, 6.4))
    fig.text(0.035, 0.96, "What S1 changed  (and what it did not)", ha="left", va="top", fontsize=13)
    fig.text(
        0.035,
        0.915,
        "n=100 · Math-Verify · temp 0.6 / top_p 0.95 / top_k 20 · A40",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )

    for i, (label, value, note) in enumerate(cards):
        ax = fig.add_axes([0.035 + i * 0.24, 0.70, 0.225, 0.18])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=0.9,
                facecolor=PANEL,
                edgecolor=RULE,
                transform=ax.transAxes,
            )
        )
        ax.text(0.5, 0.78, label, ha="center", va="center", fontsize=8, color=MUTED)
        ax.text(0.5, 0.46, value, ha="center", va="center", fontsize=12, color=S1_C if i < 2 else INK)
        ax.text(0.5, 0.18, note, ha="center", va="center", fontsize=7.5, color=MUTED)

    ax = fig.add_axes([0.07, 0.10, 0.90, 0.52])
    x = list(range(len(pairs)))
    w = 0.34
    ax.bar([i - w / 2 for i in x], [p[1] for p in pairs], w, label="Base", color=BASE_BAR, edgecolor="none")
    ax.bar([i + w / 2 for i in x], [p[2] for p in pairs], w, label="S1 @300", color=S1_C, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("percent")
    ax.yaxis.grid(True, color=RULE, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    for i, (_, bv, sv) in enumerate(pairs):
        ax.text(i - w / 2, bv + 2, f"{bv:.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(i + w / 2, sv + 2, f"{sv:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    fig.savefig(CARD / "metrics.png", dpi=180)
    plt.close(fig)


def sweep():
    models = ["base", "v4@100", "v4@200", "v4@300", "v4@400"]
    steps = [_step(m) for m in models]

    def series(set_, mode, key):
        return [100 * _row(m, set_, mode)[key] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharex=True)

    ax = axes[0]
    ax.plot(steps, series("gsm8k", "think", "closed"), marker="o", color=S1_C, lw=1.7, ms=6, label="GSM8K closed")
    ax.plot(steps, series("math500", "think", "closed"), marker="o", color="#93c5fd", lw=1.7, ms=6, label="MATH-500 closed")
    ax.plot(steps, series("gsm8k", "think", "acc"), marker="o", color=INK, lw=1.3, ms=5.5, ls="--", label="GSM8K acc")
    ax.axvline(300, color=S1_C, ls=":", lw=1.1)
    ax.text(300, 104, "ship @300", ha="center", va="bottom", fontsize=8, color=S1_C)
    ax.set_ylim(0, 112)
    ax.set_title("Think-mode protocol vs accuracy")
    ax.set_ylabel("percent")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(steps, series("gsm8k", "think", "trunc"), marker="o", color=INK, lw=1.7, ms=6, label="GSM8K trunc")
    ax.plot(steps, series("math500", "think", "trunc"), marker="o", color=S1_C, lw=1.7, ms=6, label="MATH-500 trunc")
    ax.axvline(300, color=S1_C, ls=":", lw=1.1)
    ax.text(300, 35.6, "ship @300", ha="center", va="bottom", fontsize=8, color=S1_C)
    ax.set_ylim(0, 38)
    ax.set_title("Think-mode truncation (hits 4096)")
    ax.set_ylabel("percent")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in axes:
        ax.set_xticks(steps)
        ax.set_xticklabels(["base", "100", "200", "300", "400"])
        ax.set_xlabel("SFT step")
        ax.yaxis.grid(True, color=RULE, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(CARD / "sweep.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    protocol()
    metrics()
    sweep()
    for name in ("protocol.png", "metrics.png", "sweep.png"):
        print("wrote", CARD / name, (CARD / name).stat().st_size, "bytes")
