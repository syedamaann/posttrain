"""Trainer-adapter tests. No GPU, no network, runs in seconds.

Run: python s3/test_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import score_completion, shape_group
import train

train.score_completion = score_completion
train.shape_group = shape_group

PASS = "```python\ndef add(a, b):\n    return a + b\n```"
FAIL = "```python\ndef add(a, b):\n    return 0\n```"


class _Result:
    def __init__(self, ok):
        self.outcome = "pass" if ok else "fail"
        self.reason = ""
        self.reward = 1.0 if ok else 0.0
        self.n_passed = 1 if ok else 0
        self.n_total = 1


def _run(code, tests):
    return _Result("return a + b" in code)


def test_system_text_matches_the_grade():
    marker = "`" * 3
    assert train.system_text() == (
        "Reason about the problem, then write </think>. "
        "After that, put the final solution in one Python block that opens with "
        + marker + "python."
    )


def test_chat_is_system_then_question():
    msgs = train.chat_messages("write add")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "write add"


def test_completion_text_reads_a_chat_list():
    assert train.completion_text([{"role": "assistant", "content": "hi"}]) == "hi"
    assert train.completion_text("hi") == "hi"


def test_mixed_group_is_kept():
    completions = [PASS] + [FAIL] * 7
    out = train.code_rewards(completions, ["p"] * 8, [["assert True"]] * 8, _run)
    assert out[0] == 1.0
    assert out[1:] == [0.0] * 7


def test_all_fail_group_is_zeroed():
    out = train.code_rewards([FAIL] * 8, ["p"] * 8, [["assert True"]] * 8, _run)
    assert out == [0.0] * 8


def test_all_pass_group_is_zeroed():
    out = train.code_rewards([PASS] * 8, ["p"] * 8, [["assert True"]] * 8, _run)
    assert out == [0.0] * 8


def test_groups_are_shaped_separately():
    """An all-fail group must not borrow a pass from the next group."""
    completions = [FAIL] * 8 + [PASS] + [FAIL] * 7
    ids = ["a"] * 8 + ["b"] * 8
    tests = [["assert True"]] * 16
    out = train.code_rewards(completions, ids, tests, _run)
    assert out[:8] == [0.0] * 8
    assert out[8] == 1.0


def test_misaligned_columns_raise():
    try:
        train.code_rewards([PASS], ["p", "q"], [["assert True"]], _run, group_size=1)
    except RuntimeError as exc:
        assert "misaligned" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


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
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
