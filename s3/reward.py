"""Outcome reward for S3 coding RL.

The paper's coding reward is binary: the last fenced Python block, executed
against a fixed test subsample, scores 1 only if every selected test passes.
This module adds two controls the 1.7B run needs and the paper's 12B-active
run did not:

  * a source scan for the one assert-hack the sandbox cannot see
  * the Reasoning RL group window, (0, 0.8], including dropping all-fail

Both are deviations. The card should say so. Missing a fence scores 0, which
is a parse precondition, not a format bonus.
"""

from __future__ import annotations

import hashlib
import random
import re

DAPO_HI = 0.8
TEST_CAP = 50

_PY_FENCE = re.compile(r"```(?:python3?|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_EQ = re.compile(r"def\s+__eq__\s*\(")
_EXIT = re.compile(r"\b(?:sys\.exit|os\._exit|atexit\.register)\b")
_BUILTINS = re.compile(r"builtins\.\w+\s*=")


def extract_code(text: str) -> str | None:
    """Last fenced python/py/python3 block. Drafts inside <think> lose."""
    found = _PY_FENCE.findall(text)
    if not found:
        return None
    code = found[-1].strip("\n")
    if not code.strip():
        return None
    return code


def format_diagnostics(text: str) -> dict:
    return {
        "fenced_any": "```" in text,
        "fenced_python": extract_code(text) is not None,
        "boxed_leak": "\\boxed{" in text,
    }


def subsample_tests(problem_id: str, tests: list[str], cap: int = TEST_CAP) -> list[str]:
    """Same problem always grades the same tests, in original order.

    Dedup first. EvolveCoder repeats assertions, and a repeated assert would
    otherwise inflate the count the cap is supposed to bound. The seed is the
    SHA-256 of the problem id, so the subsample does not depend on which
    rollout is being scored.
    """
    seen: set[str] = set()
    uniq: list[str] = []
    for test in tests:
        if test in seen:
            continue
        seen.add(test)
        uniq.append(test)
    if len(uniq) <= cap:
        return uniq
    seed = int.from_bytes(hashlib.sha256(problem_id.encode()).digest()[:8], "big")
    chosen = sorted(random.Random(seed).sample(range(len(uniq)), cap))
    return [uniq[i] for i in chosen]


def scan_hacks(code: str) -> list[str]:
    """Tags the reward must not trust the sandbox to catch.

    `always_equal` is the documented gap: an object whose __eq__ returns True
    satisfies equality asserts, and the canary (`assert False`) still fails.
    The other tags are cheap source signals for counters. A real `a == b` is
    not a tag.
    """
    tags: list[str] = []
    if _EQ.search(code) and "return True" in code:
        tags.append("always_equal")
    if _EXIT.search(code):
        tags.append("early_exit")
    if "tests.json" in code or _BUILTINS.search(code):
        tags.append("grader_tamper")
    return tags


def score_completion(text: str, problem_id: str, tests: list[str], run) -> dict:
    """Score one completion. `run` is the sandbox, injected so tests can stub it."""
    diag = format_diagnostics(text)
    code = extract_code(text)
    base = {**diag, "outcome": "fail", "reason": "", "reward": 0.0}
    if code is None:
        return {**base, "reason": "no_fenced_python"}
    hacks = scan_hacks(code)
    if hacks:
        return {**base, "reason": "hack:" + ",".join(hacks)}
    result = run(code, subsample_tests(problem_id, tests))
    return {
        **diag,
        "outcome": result.outcome,
        "reason": result.reason,
        "reward": result.reward,
        "n_passed": result.n_passed,
        "n_total": result.n_total,
    }


def dapo_keep(mean_r: float, hi: float = DAPO_HI) -> bool:
    return 0.0 < mean_r <= hi


def shape_group(rewards: list[float | None], hi: float = DAPO_HI) -> list[float | None]:
    """Zero a group whose scored mean is outside (0, hi]. Aborts stay None.

    The mean ignores None. Counting an infrastructure abort as 0 would keep
    an all-pass group and treat a sample the policy did not finish as a failure.
    """
    scored = [r for r in rewards if r is not None]
    if not scored:
        return list(rewards)
    if not dapo_keep(sum(scored) / len(scored), hi):
        return [None if r is None else 0.0 for r in rewards]
    return list(rewards)
