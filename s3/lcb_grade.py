"""Grade one LiveCodeBench solution. Not the training reward.

LiveCodeBench v6 is the frozen exam. Its tests are stdin/stdout or a function
call, and the private tests are a compressed blob. The assert sandbox cannot
score either. This module only grades. It does not generate, and nothing it
returns is a training reward.

A solution passes when every public and private test passes. One miss is a
fail. Comparison follows the exam: stripped lines, then a decimal compare
when a line is numeric, and a function may return a tuple where the test
expects a list.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
import zlib
from decimal import Decimal
from pathlib import Path

CPU_SECONDS = 6
WALL_PER_TEST = 6

_STDIN_RUNNER = r'''
import resource, runpy, sys

def _limit(name, value):
    try:
        resource.setrlimit(name, (value, value))
    except (ValueError, OSError):
        pass

_limit(resource.RLIMIT_CPU, {cpu})
_limit(resource.RLIMIT_AS, 2 * 1024 ** 3)
_limit(resource.RLIMIT_FSIZE, 8 * 1024 ** 2)
_limit(resource.RLIMIT_NPROC, 64)
sys.argv = ["solution.py"]
runpy.run_path("solution.py", run_name="__main__")
'''

_CALL_RUNNER = r'''
import json, os, resource, sys

def _limit(name, value):
    try:
        resource.setrlimit(name, (value, value))
    except (ValueError, OSError):
        pass

_limit(resource.RLIMIT_CPU, {cpu})
_limit(resource.RLIMIT_AS, 2 * 1024 ** 3)
_limit(resource.RLIMIT_FSIZE, 8 * 1024 ** 2)
_limit(resource.RLIMIT_NPROC, 64)

_open, _loads, _dumps, _write = open, json.loads, json.dumps, os.write
report_path, cases_path = sys.argv[1], sys.argv[2]
sys.argv = ["solution.py"]
with _open(cases_path) as fh:
    cases = _loads(fh.read())
os.unlink(cases_path)
with _open("solution.py") as fh:
    src = fh.read()
ns = {{}}
status, values = "ok", []
try:
    exec(compile(src, "solution.py", "exec"), ns)
    fn = ns["Solution"]().{fn_name} if "class Solution" in src else ns[{fn_name!r}]
    for raw in cases:
        args = [_loads(line) for line in raw.rstrip("\n").split("\n") if line != ""]
        value = fn(*args)
        if isinstance(value, tuple):
            value = list(value)
        values.append(value)
except BaseException as exc:
    status = type(exc).__name__
payload = _dumps({{"status": status, "values": values}}).encode()
fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
_write(fd, payload)
os.close(fd)
'''


def translate_private_test_cases(encoded: str) -> list[dict]:
    """Undo the lite dataset's private-test blob: base64, zlib, pickle, JSON."""
    if not encoded or not str(encoded).strip():
        return []
    decoded = base64.b64decode(encoded)
    decompressed = zlib.decompress(decoded)
    original = pickle.loads(decompressed)
    cases = json.loads(original)
    if not isinstance(cases, list):
        raise ValueError("private tests are not a list")
    return cases


def encode_private_test_cases(cases: list[dict]) -> str:
    """Test helper. The dataset already stores this encoding."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


def parse_tests(public_test_cases: str, private_test_cases: str) -> tuple[list[str], list[str]]:
    public = json.loads(public_test_cases) if public_test_cases else []
    private = translate_private_test_cases(private_test_cases)
    cases = list(public) + list(private)
    return [c["input"] for c in cases], [c["output"] for c in cases]


def fn_name_of(metadata: str | dict | None) -> str | None:
    if not metadata:
        return None
    meta = json.loads(metadata) if isinstance(metadata, str) else metadata
    name = meta.get("func_name")
    return name or None


def extract_code(text: str) -> str | None:
    """Last fenced block, matching the exam extractor. Empty if the fence is open."""
    lines = text.split("\n")
    marks = [i for i, line in enumerate(lines) if "```" in line]
    if len(marks) < 2:
        return None
    body = "\n".join(lines[marks[-2] + 1 : marks[-1]]).strip("\n")
    return body or None


def _stripped_lines(val: str) -> list[str]:
    return [line.strip() for line in val.strip().split("\n")]


def _decimals(line: str) -> list[Decimal] | None:
    try:
        return [Decimal(part) for part in line.split()]
    except Exception:
        return None


def outputs_match(prediction: str, expected: str) -> bool:
    got, want = _stripped_lines(prediction), _stripped_lines(expected)
    if len(got) != len(want):
        return False
    for left, right in zip(got, want):
        if left == right:
            continue
        dec_left, dec_right = _decimals(left), _decimals(right)
        if dec_left is None or dec_right is None or dec_left != dec_right:
            return False
    return True


def _run(cmd: list[str], cwd: str, timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=None if stdin is None else stdin.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _grade_stdin(code: str, inputs: list[str], outputs: list[str], python: str) -> tuple[int, str]:
    passed = 0
    with tempfile.TemporaryDirectory(prefix="lcb-") as work:
        Path(work, "solution.py").write_text(code)
        Path(work, "runner.py").write_text(_STDIN_RUNNER.format(cpu=CPU_SECONDS * max(len(inputs), 1)))
        for raw_in, raw_out in zip(inputs, outputs):
            try:
                proc = _run([python, "runner.py"], work, WALL_PER_TEST, stdin=raw_in)
            except subprocess.TimeoutExpired:
                return passed, "timeout"
            if proc.returncode != 0:
                return passed, "runtime"
            if not outputs_match(proc.stdout.decode(errors="replace"), raw_out):
                return passed, "wrong"
            passed += 1
    return passed, ""


def _grade_call(code: str, fn_name: str, inputs: list[str], outputs: list[str], python: str) -> tuple[int, str]:
    expected = []
    for raw in outputs:
        try:
            expected.append(json.loads(raw))
        except json.JSONDecodeError:
            return 0, "bad_expected"
    with tempfile.TemporaryDirectory(prefix="lcb-") as work:
        work_path = Path(work)
        (work_path / "solution.py").write_text(code)
        cases_path = work_path / "cases.json"
        report_path = work_path / "report.json"
        cases_path.write_text(json.dumps(inputs))
        runner = _CALL_RUNNER.format(cpu=CPU_SECONDS * max(len(inputs), 1), fn_name=fn_name)
        (work_path / "runner.py").write_text(runner)
        try:
            proc = _run(
                [python, "runner.py", str(report_path), str(cases_path)],
                work,
                WALL_PER_TEST * max(len(inputs), 1) + 5,
            )
        except subprocess.TimeoutExpired:
            return 0, "timeout"
        if not report_path.exists():
            return 0, "killed" if proc.returncode != 0 else "no_report"
        report = json.loads(report_path.read_text())
    if report["status"] != "ok":
        return 0, "load_error:" + report["status"]
    values = report["values"]
    passed = 0
    for got, want in zip(values, expected):
        if got != want:
            return passed, "wrong"
        passed += 1
    if len(values) != len(expected):
        return passed, "wrong"
    return passed, ""


def grade_solution(
    code: str | None,
    public_test_cases: str,
    private_test_cases: str,
    metadata: str | dict | None = None,
    *,
    python: str | None = None,
) -> dict:
    """Score one solution. `passed` is true only when every test passes."""
    python = python or sys.executable
    inputs, outputs = parse_tests(public_test_cases, private_test_cases)
    if not inputs:
        return {"passed": False, "n_passed": 0, "n_total": 0, "reason": "no_tests"}
    if not code or not code.strip():
        return {"passed": False, "n_passed": 0, "n_total": len(inputs), "reason": "no_code"}
    fn_name = fn_name_of(metadata)
    if fn_name:
        n_passed, reason = _grade_call(code, fn_name, inputs, outputs, python)
    else:
        n_passed, reason = _grade_stdin(code, inputs, outputs, python)
    return {
        "passed": n_passed == len(inputs) and reason == "",
        "n_passed": n_passed,
        "n_total": len(inputs),
        "reason": reason,
    }
