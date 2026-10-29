"""Assembly filters for the coding-RL prompt set.

Three checks from the paper, before any model is involved:

  * schema: a row with no asserts cannot train anything
  * dedup: the same statement under two ids is one problem
  * leakage: a training statement that matches a frozen exam problem is dropped

The exam match is the paper's pair of tests. A 400-character prefix hash, or
at least five shared character 60-grams. Difficulty filtering needs the S2
checkpoint, so it waits for the GPU.
"""

from __future__ import annotations

import hashlib
import re

PREFIX = 400
GRAM = 60
GRAM_HITS = 5
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def statement_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def prefix_hash(text: str) -> str:
    return hashlib.sha256(normalize(text)[:PREFIX].encode()).hexdigest()


def char_grams(text: str, n: int = GRAM) -> set[str]:
    body = normalize(text)
    if len(body) < n:
        return set()
    return {body[i : i + n] for i in range(len(body) - n + 1)}


def leaks(question: str, exam: list[dict]) -> str | None:
    """Return 'prefix' or 'ngram' when `question` overlaps a frozen exam row."""
    pre = prefix_hash(question)
    grams = char_grams(question)
    for row in exam:
        if prefix_hash(row["question"]) == pre:
            return "prefix"
        if len(grams & char_grams(row["question"])) >= GRAM_HITS:
            return "ngram"
    return None


def keep_row(row: dict, seen: set[str], exam: list[dict]) -> tuple[bool, str]:
    tests = row.get("tests") or []
    if not isinstance(tests, list) or not any(str(t).strip() for t in tests):
        return False, "empty_tests"
    key = statement_key(row["question"])
    if key in seen:
        return False, "duplicate"
    reason = leaks(row["question"], exam)
    if reason is not None:
        return False, f"leak_{reason}"
    seen.add(key)
    return True, "kept"


def apply_filters(rows: list[dict], exam: list[dict]) -> tuple[list[dict], dict[str, int]]:
    seen: set[str] = set()
    kept: list[dict] = []
    counts = {"started": len(rows), "kept": 0, "empty_tests": 0, "duplicate": 0, "leak_prefix": 0, "leak_ngram": 0}
    for row in rows:
        ok, reason = keep_row(row, seen, exam)
        counts[reason] = counts.get(reason, 0) + 1
        if ok:
            kept.append(row)
    counts["kept"] = len(kept)
    return kept, counts


def _demo() -> None:
    # Longer than 400 characters, so a different ending does not change the prefix.
    frozen = "do not train on this frozen exam problem. " * 12
    shared = "palindrome-appears-twice-without-overlap-in-this-exact-phrase!!"
    exam = [{"id": "exam-1", "question": frozen + shared}]
    rows = [
        {"id": "keep", "question": "Write add(a, b) and return the sum of the two integers.", "tests": ["assert add(1, 2) == 3"]},
        {"id": "empty", "question": "This problem has no tests, so it cannot score a completion.", "tests": []},
        {"id": "dup", "question": "Write add(a, b) and return the sum of the two integers.", "tests": ["assert add(0, 0) == 0"]},
        {"id": "prefix", "question": frozen + "a different ending that does not save it.", "tests": ["assert f() == 1"]},
        {"id": "grams", "question": ("unrelated opening. " * 30) + shared, "tests": ["assert f() == 1"]},
    ]
    kept, counts = apply_filters(rows, exam)
    print("kept", [row["id"] for row in kept])
    for name in ("started", "empty_tests", "duplicate", "leak_prefix", "leak_ngram", "kept"):
        print(f"{name} {counts[name]}")


if __name__ == "__main__":
    _demo()
