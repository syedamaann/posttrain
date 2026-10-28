"""Single-shot Python execution for S3 coding RL.

Stage 3 of Rufus-Air needs one thing from a sandbox: run a candidate program
against a list of `assert` statements and report pass/fail. No persistent
working tree, no multi-turn state. That is a much lower bar than the
Firecracker/E2B stack the paper describes in 4.2, which exists for the
Coding Agent stage.

Three outcomes, and the distinction between the last two is load-bearing:

    PASS    every selected test passed            -> reward 1
    FAIL    wrong answer, exception, timeout,     -> reward 0
            non-termination, tampering
    ABORT   our harness broke, not the program    -> reward None (sample dropped)

ABORT exists because the paper tags infrastructure failures separately and
removes them from the group before advantages are computed (3.6). A sample
that failed because our temp dir was full is not evidence that the policy
wrote bad code, and letting it score 0 poisons the group-relative baseline.
TRL lets a reward function return None to skip a sample, which is exactly
this.

GRADER INTEGRITY
----------------
The candidate and the grader share one interpreter, so the grader has to be
written as if the candidate is hostile. Four structural defences, each of
which closed an exploit that worked against an earlier draft of this file:

  1. Tests are read and then unlinked *before* the candidate is executed.
     Otherwise the candidate just overwrites tests.json with [].
  2. The report goes to a file whose path arrives on argv, and argv is
     cleared before the candidate runs. Otherwise the candidate registers an
     atexit hook that prints a forged sentinel after the real one.
  3. The runner calls os._exit, so candidate atexit hooks never run at all.
  4. print / open / compile / json.dumps are bound to locals before the
     candidate can rebind them in builtins.

A canary assertion that must fail is appended to every payload. It is cheap
insurance against a future refactor silently disabling assertions; it does
*not* catch an always-equal return value, which is why reward.py also scans
source for that family. Layered, not airtight.

Defence model: the pod is disposable and holds nothing of value. These limits
stop accidents and cheap exploits, not a determined attacker. Training and
baseline eval call `modal_sandbox.run_tests` instead. This local runner stays
so the reward unit tests can execute code without a Modal account.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# Resource ceilings for the child process.
CPU_SECONDS = 10  # RLIMIT_CPU: SIGKILL after this much CPU time
WALL_SECONDS = 15  # subprocess timeout: catches sleep() and blocking I/O
ADDRESS_SPACE = 2 * 1024**3  # RLIMIT_AS: 2 GiB
MAX_FILE_BYTES = 8 * 1024**2  # RLIMIT_FSIZE: stops disk-filling
MAX_PROCS = 64  # RLIMIT_NPROC: stops fork bombs
OUTPUT_CAP = 16_000  # chars of stdout/stderr we keep for logs

# A test that must fail. If it ever passes, assertions are not working and
# every other "pass" in that run is worthless.
CANARY = "assert False, 'canary'"

RUNNER = r'''
import json, os, resource, sys

def _limit(name, value):
    # macOS reports RLIMIT_AS as unlimited and then refuses every attempt to
    # lower it. A refused ceiling must not kill the grader: that turns every
    # candidate into a false "no_report" failure. On Linux the same call sticks.
    try:
        resource.setrlimit(name, (value, value))
    except (ValueError, OSError):
        pass

_limit(resource.RLIMIT_CPU, {cpu})
_limit(resource.RLIMIT_AS, {mem})
_limit(resource.RLIMIT_FSIZE, {fsize})
_limit(resource.RLIMIT_NPROC, {nproc})


def main():
    # Bind everything the grader needs before the candidate can rebind it.
    _open, _compile, _dumps, _write = open, compile, json.dumps, os.write

    report_path = sys.argv[1]
    sys.argv = ["solution.py"]          # candidate must not see the path

    with _open("tests.json") as fh:
        tests = json.load(fh)
    os.unlink("tests.json")             # candidate must not rewrite the tests

    with _open("solution.py") as fh:
        src = fh.read()

    # Advisory, not a security boundary: raises the cost of casual
    # exfiltration and of shelling out to something that confuses the grader.
    import socket
    def _blocked(*a, **k):
        raise OSError("disabled in sandbox")
    for _n in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
        setattr(socket, _n, _blocked)
    import subprocess as _sp
    _sp.Popen = _sp.run = _blocked
    os.system = _blocked
    for _n in ("fork", "forkpty", "execv", "execve"):
        if hasattr(os, _n):
            setattr(os, _n, _blocked)

    ns, status = {{}}, "ok"
    try:
        exec(_compile(src, "solution.py", "exec"), ns)
    except BaseException as exc:        # BaseException: SystemExit counts too
        status = type(exc).__name__

    results = []
    if status == "ok":
        for t in tests:
            try:
                exec(_compile(t, "<test>", "exec"), ns)
                results.append(True)
            except BaseException:
                results.append(False)

    payload = _dumps({{"status": status, "results": results}}).encode()
    fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    _write(fd, payload)
    os.close(fd)


try:
    main()
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                         # candidate atexit hooks never run
'''


@dataclass
class ExecResult:
    """Outcome of running one candidate against one test list."""

    outcome: str  # "pass" | "fail" | "abort"
    n_passed: int = 0
    n_total: int = 0
    reason: str = ""  # load_error / timeout / canary_broken / ...
    stdout: str = ""
    stderr: str = ""
    per_test: list[bool] = field(default_factory=list)

    @property
    def reward(self) -> float | None:
        """Binary, per 3.3. None means drop the sample from its group."""
        if self.outcome == "abort":
            return None
        return 1.0 if self.outcome == "pass" else 0.0


def interpret_report(
    raw: str | None,
    n_tests: int,
    *,
    stdout: str = "",
    stderr: str = "",
    missing: bool = False,
    killed: bool = False,
) -> ExecResult:
    """Turn a grader report into pass, fail, or abort. Shared by both runners."""
    common = dict(stdout=(stdout or "")[-OUTPUT_CAP:], stderr=(stderr or "")[-OUTPUT_CAP:])
    if missing or raw is None:
        reason = "killed" if killed else "no_report"
        return ExecResult("fail", n_total=n_tests, reason=reason, **common)
    try:
        report = json.loads(raw)
        results = list(report["results"])
        status = report["status"]
    except (ValueError, KeyError, TypeError):
        return ExecResult("abort", reason="bad_report", **common)
    if status != "ok":
        return ExecResult("fail", n_total=n_tests, reason=f"load_error:{status}", **common)
    if len(results) != n_tests + 1:
        return ExecResult("abort", reason="length_mismatch", **common)
    *real, canary_passed = results
    if canary_passed:
        return ExecResult("fail", n_total=n_tests, reason="canary_broken", **common)
    n_passed = sum(real)
    outcome = "pass" if n_passed == len(real) else "fail"
    return ExecResult(outcome, n_passed=n_passed, n_total=len(real), per_test=real, **common)


def run_tests(
    code: str,
    tests: list[str],
    *,
    cpu_seconds: int = CPU_SECONDS,
    wall_seconds: int = WALL_SECONDS,
    python: str | None = None,
) -> ExecResult:
    """Run `code` against `tests`, returning pass/fail/abort."""
    if not tests:
        # No test payload is a data bug, not a policy failure. The paper drops
        # these at schema validation; if one reaches here, do not score it.
        return ExecResult("abort", reason="no_tests")

    python = python or sys.executable
    # Two directories: the candidate only ever sees `workdir`.
    outdir = tempfile.mkdtemp(prefix="s3out-")
    workdir = tempfile.mkdtemp(prefix="s3exec-")
    try:
        payload = list(tests) + [CANARY]
        report_path = os.path.join(outdir, "report.json")
        runner = RUNNER.format(
            cpu=cpu_seconds,
            mem=ADDRESS_SPACE,
            fsize=MAX_FILE_BYTES,
            nproc=MAX_PROCS,
        )
        for name, body in (
            ("solution.py", code),
            ("tests.json", json.dumps(payload)),
            ("runner.py", runner),
        ):
            with open(os.path.join(workdir, name), "w") as fh:
                fh.write(body)

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        try:
            proc = subprocess.run(
                [python, "-I", "-B", "runner.py", report_path],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=wall_seconds,
            )
        except subprocess.TimeoutExpired:
            # Non-termination is a wrong answer, not an infrastructure fault.
            return ExecResult("fail", n_total=len(tests), reason="timeout")
        except (OSError, ValueError) as exc:
            # We could not even start the child. That is ours, not the policy's.
            return ExecResult("abort", reason=f"spawn_error:{type(exc).__name__}")

        if not os.path.exists(report_path):
            # The grader never reported. Either the candidate killed the
            # interpreter (os._exit, SIGKILL from RLIMIT_CPU, segfault in a C
            # extension) or it exhausted memory. All of those are failures.
            killed = bool(proc.returncode and proc.returncode < 0)
            return interpret_report(
                None, len(tests), stdout=proc.stdout or "", stderr=proc.stderr or "", missing=True, killed=killed,
            )
        try:
            with open(report_path) as fh:
                raw = fh.read()
        except OSError:
            return ExecResult("abort", reason="bad_report", stdout=(proc.stdout or "")[-OUTPUT_CAP:], stderr=(proc.stderr or "")[-OUTPUT_CAP:])
        return interpret_report(raw, len(tests), stdout=proc.stdout or "", stderr=proc.stderr or "")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
