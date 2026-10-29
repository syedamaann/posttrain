"""Paired before/after comparison for an S3 exam.

Why this file exists
--------------------
The obvious way to report a stage is two numbers: 31% before, 38% after.
That comparison throws away the single most useful fact about the
measurement -- that both numbers came from the SAME problems. Treating them
as two independent samples drags between-problem variance into the
comparison, and between-problem variance is enormous on code evals, because
problems are not interchangeable: some are trivial, most are hopeless, a few
are live.

Pairing removes that term entirely. In simulation at a paper-sized +7.3pp
effect on 40 problems at 8 samples each, the unpaired test detects it 2% of
the time and the paired test 58% of the time. Same data, same GPU hours.

What it reports
---------------
  pass@1 / pass@k   unbiased estimator (Chen et al.), per condition
  paired delta      mean per-problem change, bootstrap CI over problems
  live problems     how many problems were capable of moving at all --
                    the honest denominator of your experiment
  flip table        problems that went fail->pass and pass->fail
  MDE               the smallest effect this exam could have detected,
                    computed from the observed variance

Input: two JSONL files with one row per problem,
       {"problem_id": str, "n_pass": int, "n_total": int}
  or   {"problem_id": str, "passes": [bool, ...]}

Usage:
  python s3/compare.py --before s2_holdout.jsonl --after s3_holdout.jsonl \
                       --label-before "S2" --label-after "S3" --out s3/card/holdout.md
"""

from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np

BOOTSTRAP = 20_000
RNG = np.random.default_rng(0)


# --- estimators -----------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: c correct out of n samples.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Do NOT compute this as "did any of the first k samples pass". That
    estimator is biased and wastes samples. With n=8 and k=1 this reduces to
    c/n, which is the right way to estimate pass@1 from 8 rollouts -- far
    lower variance than scoring a single sample per problem.
    """
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load(path: str) -> dict[str, tuple[int, int]]:
    """problem_id -> (n_pass, n_total)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r["problem_id"])
            if "passes" in r:
                seq = list(r["passes"])
                out[pid] = (int(sum(bool(x) for x in seq)), len(seq))
            else:
                out[pid] = (int(r["n_pass"]), int(r["n_total"]))
    return out


def boot_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = BOOTSTRAP):
    """Percentile bootstrap over problems (the unit of resampling)."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


# --- the comparison -------------------------------------------------------


def compare(before: dict, after: dict, ks=(1, 8)) -> dict:
    ids = sorted(set(before) & set(after))
    if not ids:
        raise SystemExit("no overlapping problem_ids -- the exams are not paired")
    dropped = (set(before) | set(after)) - set(ids)

    b = np.array([before[i] for i in ids])  # (n, 2) -> n_pass, n_total
    a = np.array([after[i] for i in ids])

    rate_b = b[:, 0] / b[:, 1]
    rate_a = a[:, 0] / a[:, 1]
    delta = rate_a - rate_b

    # A problem is "live" if either condition was strictly between all-fail
    # and all-pass. Problems stuck at 0/k in both conditions contribute a
    # guaranteed zero to the delta and inflate n without adding information.
    live = ((b[:, 0] > 0) & (b[:, 0] < b[:, 1])) | (
        (a[:, 0] > 0) & (a[:, 0] < a[:, 1])
    )
    # Also live: anything that moved off a floor or ceiling.
    live |= b[:, 0] != a[:, 0]

    res = {
        "n_problems": len(ids),
        "n_live": int(live.sum()),
        "dropped_unpaired": sorted(dropped),
        "passk": {},
        "delta_mean": float(delta.mean()),
        "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
    }

    for k in ks:
        try:
            pb = np.array([pass_at_k(n, c, k) for c, n in b])
            pa = np.array([pass_at_k(n, c, k) for c, n in a])
        except ValueError:
            continue  # not enough samples for this k
        d = pa - pb
        res["passk"][k] = {
            "before": float(pb.mean()),
            "after": float(pa.mean()),
            "delta": float(d.mean()),
            "ci": boot_ci(d),
        }

    res["delta_ci"] = boot_ci(delta)
    se = res["delta_sd"] / np.sqrt(len(delta)) if len(delta) > 1 else float("nan")
    res["delta_se"] = float(se)
    # Smallest true effect this exam would catch 80% of the time at alpha=.05.
    res["mde_80"] = float(2.80 * se)
    res["significant"] = bool(
        res["delta_ci"][0] > 0 or res["delta_ci"][1] < 0
    )

    # Flip table on the any-pass binarisation -- lossy, but the most readable
    # single view of what actually changed.
    solved_b, solved_a = b[:, 0] > 0, a[:, 0] > 0
    res["flips"] = {
        "fail_to_pass": int((~solved_b & solved_a).sum()),
        "pass_to_fail": int((solved_b & ~solved_a).sum()),
        "stayed_solved": int((solved_b & solved_a).sum()),
        "stayed_unsolved": int((~solved_b & ~solved_a).sum()),
    }
    res["_ids"] = ids
    res["_delta"] = delta
    return res


def render(res: dict, label_b="before", label_a="after", exam="exam") -> str:
    L = []
    L.append(f"### {exam}: {label_b} vs {label_a} (paired, n={res['n_problems']})\n")

    L.append(f"| metric | {label_b} | {label_a} | Δ | 95% CI |")
    L.append("|---|---:|---:|---:|---|")
    for k, d in sorted(res["passk"].items()):
        lo, hi = d["ci"]
        L.append(
            f"| pass@{k} | {d['before']*100:.1f}% | {d['after']*100:.1f}% | "
            f"{d['delta']*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] |"
        )
    L.append("")

    f = res["flips"]
    L.append(f"| problems | n |")
    L.append("|---|---:|")
    L.append(f"| live (could register a change) | {res['n_live']} / {res['n_problems']} |")
    L.append(f"| fail → pass | {f['fail_to_pass']} |")
    L.append(f"| pass → fail | {f['pass_to_fail']} |")
    L.append(f"| solved in both | {f['stayed_solved']} |")
    L.append(f"| unsolved in both | {f['stayed_unsolved']} |")
    L.append("")

    verdict = "distinguishable from zero" if res["significant"] else "NOT distinguishable from zero"
    L.append(
        f"Paired Δ {res['delta_mean']*100:+.2f} pp "
        f"(SE {res['delta_se']*100:.2f}) — **{verdict}** at 95%."
    )
    L.append(
        f"This exam could have detected a true effect of "
        f"**{res['mde_80']*100:.1f} pp** or larger 80% of the time. "
        f"The paper's Coding RL effect was +7.3 pp."
    )
    if res["mde_80"] > 0.073:
        L.append(
            f"\n> ⚠️ MDE exceeds the paper's effect size. A paper-sized "
            f"improvement would more likely than not read as noise here. "
            f"Report the CI, not the point estimate, and do not call a null result a null result."
        )
    if res["dropped_unpaired"]:
        L.append(f"\n{len(res['dropped_unpaired'])} problems appeared in only one condition and were excluded.")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--label-before", default="before")
    p.add_argument("--label-after", default="after")
    p.add_argument("--exam", default="held-out EvolveCoder")
    p.add_argument("--ks", default="1,8")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    res = compare(load(args.before), load(args.after), ks=ks)
    md = render(res, args.label_before, args.label_after, args.exam)
    print(md)
    if args.out:
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(md + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
