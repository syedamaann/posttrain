"""Sandbox behaviour, including the exploits a code-RL reward has to survive.

Run: python -m pytest s3/test_sandbox.py -q      (or: python s3/test_sandbox.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sandbox import run_tests

GOOD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]


def test_correct_solution_passes():
    r = run_tests(GOOD, TESTS)
    assert r.outcome == "pass", r
    assert r.reward == 1.0
    assert (r.n_passed, r.n_total) == (2, 2)


def test_wrong_solution_fails():
    r = run_tests("def add(a, b):\n    return a - b\n", TESTS)
    assert r.outcome == "fail"
    assert r.reward == 0.0
    assert r.n_passed < r.n_total


def test_partial_pass_is_still_zero():
    """Binary reward, per 3.3: all selected tests, or nothing."""
    r = run_tests("def add(a, b):\n    return abs(a) + abs(b)\n", TESTS)
    assert r.outcome == "fail"
    assert r.n_passed == 1 and r.n_total == 2
    assert r.reward == 0.0


def test_syntax_error_fails_not_aborts():
    r = run_tests("def add(a, b)\n    return a + b\n", TESTS)
    assert r.outcome == "fail"
    assert r.reason.startswith("load_error:SyntaxError")


def test_missing_name_fails():
    r = run_tests("x = 1\n", TESTS)
    assert r.outcome == "fail"
    assert r.n_passed == 0


# --- exploits -------------------------------------------------------------


def test_sys_exit_zero_does_not_score():
    """A classic: exit cleanly before anything can be checked."""
    r = run_tests("import sys\nsys.exit(0)\n", TESTS)
    assert r.outcome == "fail"
    assert "SystemExit" in r.reason


def test_os_exit_does_not_score():
    """os._exit bypasses the exception machinery entirely -> no sentinel."""
    r = run_tests("import os\nos._exit(0)\n", TESTS)
    assert r.outcome == "fail"
    assert r.reason in ("no_report", "killed")


def test_patching_builtins_assertionerror_does_not_help():
    """A tempting exploit that does not actually work: since 3.9 CPython emits
    LOAD_ASSERTION_ERROR, which reads the real exception type, not the name in
    builtins. Kept as a test so the claim stays true if that ever changes."""
    code = (
        "import builtins\n"
        "class _Never(Exception):\n    pass\n"
        "builtins.AssertionError = _Never\n"
        "def add(a, b):\n    return 999\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "fail"
    assert r.reward == 0.0


def test_overwriting_the_test_payload_is_defeated():
    """Candidate rewrites tests.json to an empty list. The grader reads and
    unlinks the payload before the candidate runs, so this is inert."""
    code = (
        "open('tests.json', 'w').write('[]')\n"
        "def add(a, b):\n    return 999\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "fail", "tests must have survived the rewrite"
    assert r.n_total == 2


def test_forged_sentinel_via_atexit_is_defeated():
    """Candidate registers an atexit hook that forges a clean report. The
    grader writes its report to a path the candidate cannot see, then calls
    os._exit, so atexit hooks never run."""
    code = (
        "import atexit, json\n"
        "def _forge():\n"
        "    print(json.dumps({'status': 'ok', 'results': [True, True, True]}))\n"
        "atexit.register(_forge)\n"
        "def add(a, b):\n    return 999\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "fail"


def test_rebinding_print_and_dumps_is_defeated():
    """Candidate replaces the functions the grader reports with. The grader
    bound them to locals before executing the candidate."""
    code = (
        "import builtins, json\n"
        "builtins.print = lambda *a, **k: None\n"
        "json.dumps = lambda *a, **k: '{\"status\":\"ok\",\"results\":[true,true,true]}'\n"
        "def add(a, b):\n    return 999\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "fail"


def test_always_equal_return_value_defeats_asserts():
    """KNOWN GAP, documented deliberately.

    An object whose __eq__ always returns True satisfies every equality
    assert, and the canary (`assert False`) still fails, so the sandbox
    cannot see it. This is the single most dangerous hack in assert-based
    code RL. It is caught one layer up, by the source scan in reward.py --
    see test_reward.test_always_equal_is_flagged."""
    code = (
        "class _Any:\n"
        "    def __eq__(self, other):\n        return True\n"
        "def add(a, b):\n    return _Any()\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "pass", "documents the gap the reward layer must close"


def test_timeout_is_failure_not_abort():
    r = run_tests("def add(a, b):\n    while True:\n        pass\n", TESTS, wall_seconds=3)
    assert r.outcome == "fail"
    assert r.reason in ("timeout", "killed", "no_sentinel")
    assert r.reward == 0.0


def test_network_is_blocked():
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('example.com', 80), timeout=1)\n"
        "    _ok = True\n"
        "except Exception:\n"
        "    _ok = False\n"
        "def add(a, b):\n    return (a + b) if not _ok else 0\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "pass", "network stub should have made _ok False"


def test_subprocess_is_blocked():
    code = (
        "import subprocess\n"
        "try:\n"
        "    subprocess.run(['echo', 'hi'])\n"
        "    _ok = True\n"
        "except Exception:\n"
        "    _ok = False\n"
        "def add(a, b):\n    return (a + b) if not _ok else 0\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "pass"


def test_reading_the_test_payload_does_not_help_by_itself():
    """The candidate can see tests.json. Reading it is not in itself a pass;
    it only helps if the model also writes code that satisfies them, which is
    the task. This documents the exposure rather than claiming it is closed."""
    code = (
        "import json, os\n"
        "_seen = os.path.exists('tests.json')\n"
        "def add(a, b):\n    return a + b\n"
    )
    r = run_tests(code, TESTS)
    assert r.outcome == "pass"


def test_empty_test_list_aborts():
    r = run_tests(GOOD, [])
    assert r.outcome == "abort"
    assert r.reward is None


def test_huge_stdout_is_capped():
    code = "for _ in range(200000):\n    print('x' * 80)\ndef add(a, b):\n    return a + b\n"
    r = run_tests(code, TESTS, wall_seconds=20)
    assert len(r.stdout) <= 20_000


def test_fork_bomb_does_not_hang_the_harness():
    code = "import os\nwhile True:\n    os.fork()\n"
    r = run_tests(code, TESTS, wall_seconds=5)
    assert r.outcome == "fail"


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
