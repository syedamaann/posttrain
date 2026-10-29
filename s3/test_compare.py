"""Tests for the paired comparison. Run: python s3/test_compare.py"""

import json
import sys
import tempfile
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from compare import compare, load, pass_at_k, render


def test_pass_at_k_matches_closed_form():
    # 8 samples, 1 correct: pass@1 = 1/8
    assert isclose(pass_at_k(8, 1, 1), 0.125)
    # all correct -> 1 at any k
    assert pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 8, 8) == 1.0
    # none correct -> 0 at any k
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 0, 8) == 0.0
    # 8 samples, 1 correct, k=8 -> certain to include it
    assert pass_at_k(8, 1, 8) == 1.0


def test_pass_at_k_is_the_unbiased_estimator_not_first_k():
    """4 of 8 correct. pass@2 should be 1 - C(4,2)/C(8,2) = 1 - 6/28."""
    assert isclose(pass_at_k(8, 4, 2), 1 - 6 / 28)


def test_pass_at_k_rejects_impossible_k():
    try:
        pass_at_k(4, 1, 8)
    except ValueError:
        return
    raise AssertionError("should have refused pass@8 from 4 samples")


def _write(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


def test_load_accepts_both_schemas():
    a = _write([{"problem_id": "p1", "n_pass": 3, "n_total": 8}])
    b = _write([{"problem_id": "p1", "passes": [1, 1, 1, 0, 0, 0, 0, 0]}])
    assert load(a) == load(b) == {"p1": (3, 8)}


def test_identical_inputs_give_zero_delta():
    rows = [{"problem_id": f"p{i}", "n_pass": i % 9, "n_total": 8} for i in range(30)]
    rows = [{**r, "n_pass": min(r["n_pass"], 8)} for r in rows]
    d = load(_write(rows))
    res = compare(d, d)
    assert isclose(res["delta_mean"], 0.0, abs_tol=1e-12)
    assert not res["significant"]


def test_clear_improvement_is_detected():
    before = {f"p{i}": (2, 8) for i in range(60)}
    after = {f"p{i}": (6, 8) for i in range(60)}
    res = compare(before, after)
    assert res["delta_mean"] > 0.49
    assert res["significant"]
    assert res["flips"]["fail_to_pass"] == 0  # both already "solved" (any-pass)
    assert res["flips"]["stayed_solved"] == 60


def test_flip_table_counts_correctly():
    before = {"a": (0, 8), "b": (4, 8), "c": (0, 8), "d": (5, 8)}
    after = {"a": (3, 8), "b": (0, 8), "c": (0, 8), "d": (6, 8)}
    f = compare(before, after)["flips"]
    assert f == {
        "fail_to_pass": 1,       # a
        "pass_to_fail": 1,       # b
        "stayed_solved": 1,      # d
        "stayed_unsolved": 1,    # c
    }


def test_live_count_excludes_frozen_problems():
    """Problems at 0/8 in both conditions cannot register anything."""
    before = {**{f"dead{i}": (0, 8) for i in range(20)}, "live": (3, 8)}
    after = {**{f"dead{i}": (0, 8) for i in range(20)}, "live": (5, 8)}
    res = compare(before, after)
    assert res["n_problems"] == 21
    assert res["n_live"] == 1, res["n_live"]


def test_unpaired_ids_are_dropped_not_crashed():
    before = {"a": (1, 8), "b": (2, 8)}
    after = {"a": (3, 8), "c": (4, 8)}
    res = compare(before, after)
    assert res["n_problems"] == 1
    assert set(res["dropped_unpaired"]) == {"b", "c"}


def test_no_overlap_exits_loudly():
    try:
        compare({"a": (1, 8)}, {"z": (1, 8)})
    except SystemExit:
        return
    raise AssertionError("should refuse to compare non-overlapping exams")


def _synthetic(n, k, seed, effect=0.073):
    rng = np.random.default_rng(seed)
    before, after = {}, {}
    for i in range(n):
        p = rng.beta(0.4, 2.9)  # random unfiltered EvolveCoder for a 1.7B
        before[f"p{i}"] = (int(rng.binomial(k, p)), k)
        after[f"p{i}"] = (int(rng.binomial(k, min(p + effect, 1.0))), k)
    return before, after


def test_small_exam_mde_sits_at_the_paper_effect_size():
    """n=40 at k=8 lands with MDE ~7.2pp against the paper's +7.3pp effect:
    right on the boundary, which is the point. Measured across seeds, since
    any single draw straddles it."""
    mdes = [compare(*_synthetic(40, 8, s))["mde_80"] for s in range(40)]
    median = float(np.median(mdes))
    assert 0.06 < median < 0.09, f"MDE moved unexpectedly: {median:.3f}"


def test_k4_is_meaningfully_worse_than_k8():
    """Samples per problem are not free resolution, but they are cheap."""
    m4 = float(np.median([compare(*_synthetic(40, 4, s))["mde_80"] for s in range(40)]))
    m8 = float(np.median([compare(*_synthetic(40, 8, s))["mde_80"] for s in range(40)]))
    assert m4 > m8 * 1.2, f"k=4 {m4:.3f} vs k=8 {m8:.3f}"


def test_underpowered_exam_is_flagged_in_the_report():
    """Whatever the seed, an exam whose MDE exceeds 7.3pp must say so."""
    before, after = _synthetic(25, 4, 7)
    res = compare(before, after)
    assert res["mde_80"] > 0.073
    assert "⚠️" in render(res)


def test_large_exam_is_adequately_powered():
    rng = np.random.default_rng(2)
    before, after = {}, {}
    for i in range(400):
        p = rng.beta(0.4, 2.9)
        before[f"p{i}"] = (int(rng.binomial(8, p)), 8)
        after[f"p{i}"] = (int(rng.binomial(8, min(p + 0.10, 1.0))), 8)
    res = compare(before, after)
    assert res["significant"]
    assert res["mde_80"] < 0.073


def test_render_produces_markdown_with_both_ks():
    before = {f"p{i}": (2, 8) for i in range(40)}
    after = {f"p{i}": (4, 8) for i in range(40)}
    md = render(compare(before, after), "S2", "S3")
    assert "pass@1" in md and "pass@8" in md
    assert "S2" in md and "S3" in md
    assert "95% CI" in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
