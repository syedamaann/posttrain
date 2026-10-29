"""LiveCodeBench grader tests. No GPU, no network, runs in seconds.

Run: python s3/test_lcb_grade.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lcb_grade import (
    encode_private_test_cases,
    extract_code,
    grade_solution,
    outputs_match,
    parse_tests,
    translate_private_test_cases,
)

PUBLIC_STDIN = json.dumps([
    {"input": "2\n3\n", "output": "5\n", "testtype": "stdin"},
])
ADD = "a, b = map(int, __import__('sys').stdin.read().split())\nprint(a + b)\n"


def test_private_blob_roundtrips():
    cases = [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
    assert translate_private_test_cases(encode_private_test_cases(cases)) == cases


def test_empty_private_blob_is_no_cases():
    assert translate_private_test_cases("") == []


def test_parse_joins_public_and_private():
    private = encode_private_test_cases([
        {"input": "9\n", "output": "9\n", "testtype": "stdin"},
    ])
    inputs, outputs = parse_tests(PUBLIC_STDIN, private)
    assert inputs == ["2\n3\n", "9\n"]
    assert outputs == ["5\n", "9\n"]


def test_extracts_the_last_fence():
    text = "draft\n```\nNO\n```\nfinal\n```python\nYES\n```\n"
    assert extract_code(text) == "YES"


def test_open_fence_is_no_code():
    assert extract_code("```python\nno close") is None


def test_decimal_lines_match():
    assert outputs_match("1.0 2\n", "1.00 2.0")
    assert not outputs_match("1\n2\n", "1\n")


def test_stdin_solution_passes_public_and_private():
    private = encode_private_test_cases([
        {"input": "10\n20\n", "output": "30\n", "testtype": "stdin"},
    ])
    result = grade_solution(ADD, PUBLIC_STDIN, private)
    assert result["passed"] is True, result
    assert result["n_passed"] == 2


def test_stdin_wrong_answer_fails():
    result = grade_solution("print(0)\n", PUBLIC_STDIN, "")
    assert result["passed"] is False
    assert result["reason"] == "wrong"
    assert result["n_passed"] == 0


def test_stdin_timeout_fails():
    result = grade_solution("while True:\n    pass\n", PUBLIC_STDIN, "")
    assert result["passed"] is False
    assert result["reason"] == "timeout"


def test_call_solution_passes():
    public = json.dumps([
        {"input": "1\n2\n", "output": "3", "testtype": "functional"},
    ])
    code = "def add(a, b):\n    return a + b\n"
    result = grade_solution(code, public, "", {"func_name": "add"})
    assert result["passed"] is True, result


def test_call_tuple_matches_a_list():
    public = json.dumps([
        {"input": "1\n", "output": "[1, 1]", "testtype": "functional"},
    ])
    code = "def pair(a):\n    return (a, a)\n"
    result = grade_solution(code, public, "", {"func_name": "pair"})
    assert result["passed"] is True, result


def test_class_solution_method():
    public = json.dumps([
        {"input": "1\n2\n", "output": "3", "testtype": "functional"},
    ])
    code = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
    result = grade_solution(code, public, "", json.dumps({"func_name": "add"}))
    assert result["passed"] is True, result


def test_missing_code_fails_without_running():
    result = grade_solution(None, PUBLIC_STDIN, "")
    assert result["reason"] == "no_code"
    assert result["n_total"] == 1


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
