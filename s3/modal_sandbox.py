"""Run one candidate against its asserts inside a fresh Modal sandbox.

This is the executor training and baseline eval pass to `score_completion`.
Each call creates a sandbox, writes the grader and the program into it, runs
them, and terminates the sandbox before returning. A program that fails tests
scores 0. A Modal failure that never produces a report scores None.
"""

from __future__ import annotations

import json

from sandbox import (
    ADDRESS_SPACE,
    CANARY,
    CPU_SECONDS,
    MAX_FILE_BYTES,
    MAX_PROCS,
    RUNNER,
    ExecResult,
    interpret_report,
)

APP_NAME = "rufus-s3-sandbox"
WORK = "/work"
REPORT = "/var/s3-report.json"


def run_tests(
    code: str,
    tests: list[str],
    *,
    cpu_seconds: int = CPU_SECONDS,
    wall_seconds: int = 30,
    cpu: float = 1.0,
    memory: int = 2048,
    app_name: str = APP_NAME,
) -> ExecResult:
    """Grade `code` in a new network-blocked sandbox. Same contract as sandbox.run_tests."""
    if not tests:
        return ExecResult("abort", reason="no_tests")
    try:
        import modal
    except ImportError:
        return ExecResult("abort", reason="spawn_error:ModuleNotFoundError")

    sb = None
    try:
        app = modal.App.lookup(app_name, create_if_missing=True)
        sb = modal.Sandbox.create(
            app=app,
            block_network=True,
            cpu=cpu,
            memory=memory,
            timeout=wall_seconds + 60,
            workdir=WORK,
        )
        payload = list(tests) + [CANARY]
        runner = RUNNER.format(
            cpu=cpu_seconds,
            mem=ADDRESS_SPACE,
            fsize=MAX_FILE_BYTES,
            nproc=MAX_PROCS,
        )
        sb.filesystem.make_directory(WORK)
        sb.filesystem.write_text(code, f"{WORK}/solution.py")
        sb.filesystem.write_text(json.dumps(payload), f"{WORK}/tests.json")
        sb.filesystem.write_text(runner, f"{WORK}/runner.py")
        try:
            proc = sb.exec(
                "python", "-I", "-B", f"{WORK}/runner.py", REPORT,
                workdir=WORK,
                timeout=wall_seconds,
            )
            proc.wait()
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower():
                return ExecResult("fail", n_total=len(tests), reason="timeout")
            return ExecResult("abort", reason=f"spawn_error:{type(exc).__name__}")
        stdout = _read(proc.stdout)
        stderr = _read(proc.stderr)
        try:
            raw = sb.filesystem.read_text(REPORT)
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "notfound" in name or "no such" in str(exc).lower():
                killed = bool(getattr(proc, "returncode", 0) and proc.returncode < 0)
                return interpret_report(None, len(tests), stdout=stdout, stderr=stderr, missing=True, killed=killed)
            return ExecResult("abort", reason=f"bad_report:{type(exc).__name__}", stdout=stdout[-16_000:], stderr=stderr[-16_000:])
        return interpret_report(raw, len(tests), stdout=stdout, stderr=stderr)
    except Exception as exc:
        return ExecResult("abort", reason=f"spawn_error:{type(exc).__name__}")
    finally:
        if sb is not None:
            try:
                sb.terminate()
            except Exception:
                pass


def _read(stream) -> str:
    try:
        return stream.read() or ""
    except Exception:
        return ""
