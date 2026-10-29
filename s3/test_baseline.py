"""Holdout and summary locks. No download, no GPU.

Run: python3 s3/test_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from baseline import choose_holdout, summarize, usable


def test_empty_rows_never_enter_the_holdout():
    rows = usable([
        {"id": "a", "question": "one", "tests": ["assert f() == 1"]},
        {"id": "b", "question": "two", "tests": []},
        {"id": "c", "question": "one", "tests": ["assert f() == 2"]},
    ])
    assert [r["id"] for r in rows] == ["a"]


def test_holdout_is_stable():
    rows = [{"id": f"p{i}", "question": str(i), "tests": ["assert True"]} for i in range(20)]
    a = [r["id"] for r in choose_holdout(rows, 5, seed=0)]
    b = [r["id"] for r in choose_holdout(rows, 5, seed=0)]
    c = [r["id"] for r in choose_holdout(rows, 5, seed=1)]
    assert a == b and a != c and len(a) == 5


def test_summary_is_the_baseline_table():
    records = [
        {"fenced_python": True, "boxed_leak": False, "trunc": False, "tokens": 100, "prompt_skip": False, "reward": 1.0},
        {"fenced_python": False, "boxed_leak": True, "trunc": False, "tokens": 40, "prompt_skip": False, "reward": 0.0},
        {"fenced_python": False, "boxed_leak": False, "trunc": True, "tokens": 2048, "prompt_skip": False, "reward": 0.0},
        {"fenced_python": False, "boxed_leak": False, "trunc": False, "tokens": 0, "prompt_skip": True},
    ]
    s = summarize(records)
    assert s["n"] == 4
    assert s["fenced"] == 0.25
    assert s["boxed_leak"] == 0.25
    assert s["trunc"] == 0.25
    assert s["prompt_skip"] == 1
    assert s["scored"] == 3
    assert abs(s["pass"] - (1 / 3)) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
