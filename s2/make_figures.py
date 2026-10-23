#!/usr/bin/env python3
"""Write s2/card/{protocol,metrics,leak}.png. Numbers are the frozen n=100 evals."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CARD = Path(__file__).resolve().parent / "card"

INK = "#18181b"
MUTED = "#71717a"
RULE = "#e4e4e7"
PANEL = "#fafafa"
S2_C = "#1d4ed8"
S2_FILL = "#eff6ff"
LEAK_C = "#c2410c"
CHIP = "#ffffff"
CHIP_EDGE = "#d4d4d8"
BG = "#ffffff"
BEFORE = "#a1a1aa"

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


def _chip(ax, x, y, text, *, accent=False, warn=False, fontsize=8):
    if accent:
        fc, ec, tc = S2_FILL, S2_C, S2_C
    elif warn:
        fc, ec, tc = "#fff7ed", LEAK_C, LEAK_C
    else:
        fc, ec, tc = CHIP, CHIP_EDGE, INK
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
    for i, (text, kind) in enumerate(items):
        x = _chip(ax, x, y, text, accent=(kind == "s2"), warn=(kind == "leak"))
        if i < len(items) - 1:
            ax.text(x + 0.018, y, "→", ha="center", va="center", fontsize=10, color=MUTED, zorder=3)
            x += 0.036
    return x


def _panel(ax):
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


def _card(fig, i, n, label, value, note, *, accent=True):
    ax = fig.add_axes([0.035 + i * (0.93 / n), 0.70, (0.93 / n) - 0.015, 0.18])
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
    ax.text(0.5, 0.46, value, ha="center", va="center", fontsize=13, color=S2_C if accent else INK)
    ax.text(0.5, 0.18, note, ha="center", va="center", fontsize=7.5, color=MUTED)


def _bars(ax, labels, series, ylim):
    n = len(labels)
    k = len(series)
    w = 0.8 / k
    x = list(range(n))
    for s, (name, color, values) in enumerate(series):
        offset = (s - (k - 1) / 2) * w
        drawn = [ylim * 0.02 if v == 0 else v for v in values]
        ax.bar([i + offset for i in x], drawn, w * 0.92, label=name, color=color, edgecolor="none")
        for i, v in enumerate(values):
            ax.text(i + offset, drawn[i] + ylim * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK if s == k - 1 else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("percent")
    ax.yaxis.grid(True, color=RULE, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8)


def protocol():
    fig = plt.figure(figsize=(11.4, 7.05))
    fig.text(0.5, 0.975, "No-think. One answer, not two.", ha="center", va="top", fontsize=15)
    fig.text(
        0.5,
        0.935,
        "Janet's ducks, gold 18.  The prompt already ended in an empty  </think>.",
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    ax_l = fig.add_axes([0.035, 0.07, 0.45, 0.80])
    ax_r = fig.add_axes([0.515, 0.07, 0.45, 0.80])
    _panel(ax_l)
    _panel(ax_r)
    ax_l.text(0.5, 0.94, "Before  ·  accuracy checkpoint", ha="center", va="center", fontsize=10, color=LEAK_C)
    ax_r.text(0.5, 0.94, "After  ·  this repo", ha="center", va="center", fontsize=10, color=S2_C)
    _flow(ax_l, [(r"\boxed{18}", "leak"), ("</think>", "leak"), (r"\boxed{18}", "leak")], 0.06, 0.84)
    _flow(ax_r, [(r"\boxed{18}", "s2"), ("<|im_end|>", "s2")], 0.06, 0.84)
    ax_l.text(0.06, 0.74, "Writes the answer, closes again, writes it again.", fontsize=8, color=MUTED)
    ax_r.text(0.06, 0.74, "Stops after the first answer.", fontsize=8, color=MUTED)
    mono = dict(ha="left", va="top", fontsize=8, fontfamily="DejaVu Sans Mono", linespacing=1.55, color=INK)
    ax_l.text(
        0.06,
        0.66,
        "She sells 9 eggs at $2.\n"
        "She makes \\boxed{18}.\n"
        "\n"
        "</think>\n"
        "\n"
        "She sells 9 eggs at $2.\n"
        "She makes \\boxed{18}.",
        **mono,
    )
    ax_r.text(
        0.06,
        0.66,
        "She sells 9 eggs at $2.\n"
        "She makes \\boxed{18}.<|im_end|>",
        **mono,
    )
    fig.text(
        0.5,
        0.012,
        "Real shape from the accuracy checkpoint.  GSM8K no-think extra </think>  24% → 0%.   MATH-500  16% → 0%.   n=100.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    fig.savefig(CARD / "protocol.png", dpi=180)
    plt.close(fig)


def metrics():
    fig = plt.figure(figsize=(11.4, 6.4))
    fig.text(0.035, 0.96, "What S2 changed against S1", ha="left", va="top", fontsize=13)
    fig.text(
        0.035,
        0.915,
        "n=100  ·  Math-Verify  ·  temp 0.6 / top_p 0.95 / top_k 20  ·  think@4096  ·  no-think@1024",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    cards = [
        ("GSM8K think acc", "78%  →  83%", "above the 78% bar"),
        ("MATH-500 trunc", "22%  →  11%", "think, hits 4096"),
        ("GSM8K think closed", "88%  →  99%", "</think>"),
        ("GSM8K think tokens", "1319  →  629", "shorter, still closed"),
    ]
    for i, (label, value, note) in enumerate(cards):
        _card(fig, i, 4, label, value, note)
    ax = fig.add_axes([0.07, 0.10, 0.90, 0.52])
    _bars(
        ax,
        ["GSM8K think\naccuracy", "GSM8K think\n</think> closed", "MATH-500 think\ntruncated  ↓", "MATH-500 think\n</think> closed", "GSM8K no-think\nboxed"],
        [
            ("S1 @300", BEFORE, [78, 88, 22, 77, 98]),
            ("S2  this repo", S2_C, [83, 99, 11, 89, 99]),
        ],
        120,
    )
    fig.savefig(CARD / "metrics.png", dpi=180)
    plt.close(fig)


def leak():
    fig = plt.figure(figsize=(11.4, 5.5))
    fig.text(0.035, 0.96, "The extra no-think tag came back, then left", ha="left", va="top", fontsize=13)
    fig.text(
        0.035,
        0.905,
        "Count of  </think>  in the new text.  The template has already closed an empty think.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    _card(fig, 0, 2, "GSM8K no-think", "24%  →  0%", "accuracy checkpoint → this repo", accent=True)
    _card(fig, 1, 2, "MATH-500 no-think", "16%  →  0%", "S1 was already 0% on both", accent=True)
    ax = fig.add_axes([0.07, 0.12, 0.88, 0.50])
    _bars(
        ax,
        ["GSM8K no-think\nextra </think>", "MATH-500 no-think\nextra </think>"],
        [
            ("S1 @300", BEFORE, [0, 0]),
            ("Accuracy checkpoint", LEAK_C, [24, 16]),
            ("S2  this repo", S2_C, [0, 0]),
        ],
        40,
    )
    fig.savefig(CARD / "leak.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    CARD.mkdir(parents=True, exist_ok=True)
    protocol()
    metrics()
    leak()
    for name in ("protocol.png", "metrics.png", "leak.png"):
        print("wrote", CARD / name, (CARD / name).stat().st_size, "bytes")
