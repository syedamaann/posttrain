"""Reward unit tests. No GPU, no network, runs in seconds.

Run: python s3/test_reward.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import (
    dapo_keep,
    extract_code,
    format_diagnostics,
    scan_hacks,
    score_completion,
    shape_group,
    subsample_tests,
)
from sandbox import run_tests

TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]


# --- parsing --------------------------------------------------------------


def test_extracts_fenced_python():
    text = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert extract_code(text) == "def add(a, b):\n    return a + b"


def test_takes_the_last_block_not_the_first():
    """A thinking model drafts before it commits. Grading the draft is wrong."""
    text = (
        "<think>maybe\n```python\nWRONG\n```\nno, better:</think>\n"
        "```python\nRIGHT\n```"
    )
    assert extract_code(text) == "RIGHT"


def test_accepts_py_and_python3_fences():
    assert extract_code("```py\nX\n```") == "X"
    assert extract_code("```python3\nX\n```") == "X"


def test_unlabelled_fence_is_not_python():
    """Paper grades fenced ```python. An unlabelled fence is a format miss,
    but format_diagnostics records it so we can tell the two apart."""
    text = "```\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(text) is None
    d = format_diagnostics(text)
    assert d["fenced_any"] and not d["fenced_python"]


def test_no_code_at_all():
    assert extract_code("The answer is 4.") is None


def test_empty_block_is_none():
    assert extract_code("```python\n\n```") is None


def test_boxed_leak_is_detected():
    """The S2 prior we are watching for."""
    d = format_diagnostics("The answer is \\boxed{42}")
    assert d["boxed_leak"] and not d["fenced_python"]


# --- deterministic subsampling -------------------------------------------


def test_subsample_is_stable_across_calls():
    tests = [f"assert f({i}) == {i}" for i in range(200)]
    a = subsample_tests("TACO_1", tests)
    b = subsample_tests("TACO_1", tests)
    assert a == b
    assert len(a) == 50


def test_subsample_differs_between_problems():
    tests = [f"assert f({i}) == {i}" for i in range(200)]
    assert subsample_tests("TACO_1", tests) != subsample_tests("TACO_2", tests)


def test_subsample_deduplicates_first():
    """EvolveCoder-22K repeats assertions in some problems."""
    assert subsample_tests("p", ["assert a", "assert a", "assert b"]) == [
        "assert a",
        "assert b",
    ]


def test_subsample_under_cap_is_identity_order():
    tests = ["assert a", "assert b", "assert c"]
    assert subsample_tests("p", tests) == tests


def test_subsample_preserves_original_order():
    tests = [f"assert f({i})" for i in range(200)]
    out = subsample_tests("TACO_9", tests)
    assert out == [t for t in tests if t in set(out)]


# --- hack scanning --------------------------------------------------------


def test_always_equal_is_flagged():
    """The gap the sandbox provably cannot close. Closed here instead."""
    code = "class _Any:\n    def __eq__(self, other):\n        return True\n"
    assert "always_equal" in scan_hacks(code)


def test_early_exit_is_flagged():
    assert "early_exit" in scan_hacks("import sys\nsys.exit(0)\n")
    assert "early_exit" in scan_hacks("import atexit\natexit.register(f)\n")


def test_grader_tamper_is_flagged():
    assert "grader_tamper" in scan_hacks("open('tests.json','w')")
    assert "grader_tamper" in scan_hacks("import builtins\nbuiltins.print = None")


def test_clean_solution_is_not_flagged():
    code = (
        "from collections import deque\n"
        "def solve(grid):\n"
        "    q = deque()\n"
        "    total = sum(x for row in grid for x in row)\n"
        "    return total\n"
    )
    assert scan_hacks(code) == []


def test_ordinary_equality_use_is_not_flagged():
    """Guard against the __eq__ pattern being too eager."""
    code = "def add(a, b):\n    if a == b:\n        return 2 * a\n    return a + b\n"
    assert scan_hacks(code) == []


# --- end-to-end scoring ---------------------------------------------------


def test_correct_completion_scores_one():
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    r = score_completion(text, "p1", TESTS, run_tests)
    assert r["reward"] == 1.0 and r["outcome"] == "pass"


def test_wrong_completion_scores_zero():
    text = "```python\ndef add(a, b):\n    return a * b\n```"
    r = score_completion(text, "p1", TESTS, run_tests)
    assert r["reward"] == 0.0


def test_missing_fence_scores_zero_with_reason():
    r = score_completion("The answer is \\boxed{3}", "p1", TESTS, run_tests)
    assert r["reward"] == 0.0
    assert r["reason"] == "no_fenced_python"
    assert r["boxed_leak"] is True


def test_always_equal_completion_scores_zero_end_to_end():
    """Would pass the sandbox. Must not pass the reward."""
    text = (
        "```python\nclass _Any:\n    def __eq__(self, other):\n        return True\n"
        "def add(a, b):\n    return _Any()\n```"
    )
    r = score_completion(text, "p1", TESTS, run_tests)
    assert r["reward"] == 0.0
    assert r["reason"].startswith("hack:always_equal")


def test_abort_propagates_as_none():
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    r = score_completion(text, "p1", [], run_tests)  # empty payload -> abort
    assert r["reward"] is None


# --- group shaping --------------------------------------------------------


def test_dapo_window_matches_s2():
    assert not dapo_keep(0.0)
    assert dapo_keep(0.5)
    assert dapo_keep(0.8)
    assert not dapo_keep(0.9)
    assert not dapo_keep(1.0)


def test_all_fail_group_is_zeroed():
    assert shape_group([0.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 0.0]


def test_all_pass_group_is_zeroed():
    assert shape_group([1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0]


def test_mixed_group_survives():
    assert shape_group([1.0, 0.0, 0.0, 0.0]) == [1.0, 0.0, 0.0, 0.0]


def test_aborts_excluded_from_group_mean():
    """The case that makes the ABORT/FAIL distinction worth the code.

    Four samples ran and all passed; a fifth aborted on infrastructure.
      excluding the abort : mean 1.0  -> all-pass, no gradient, group dropped
      counting it as 0    : mean 0.8  -> inside (0, 0.8], group kept

    The second is wrong twice over. It keeps a group that has nothing to
    teach, and it computes advantages in which a sample the policy never
    produced is scored as a failure."""
    out = shape_group([1.0, 1.0, 1.0, 1.0, None])
    assert out == [0.0, 0.0, 0.0, 0.0, None], out


def test_aborts_do_not_rescue_an_all_fail_group():
    out = shape_group([0.0, 0.0, None])
    assert out == [0.0, 0.0, None], out


def test_mixed_group_with_abort_survives():
    out = shape_group([1.0, 0.0, 0.0, None])
    assert out == [1.0, 0.0, 0.0, None], out


def test_all_abort_group_passes_through():
    assert shape_group([None, None]) == [None, None]


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
