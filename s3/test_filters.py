"""Filter locks. No download, no GPU.

Run: python3 s3/test_filters.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from filters import apply_filters, char_grams, leaks


FROZEN = "do not train on this frozen exam problem. " * 12
SHARED = "palindrome-appears-twice-without-overlap-in-this-exact-phrase!!"
EXAM = [{"id": "exam-1", "question": FROZEN + SHARED}]


def test_empty_duplicate_and_both_leak_rules():
    rows = [
        {"id": "keep", "question": "Write add(a, b) and return the sum.", "tests": ["assert add(1, 2) == 3"]},
        {"id": "blank", "question": "Nothing to grade.", "tests": ["   "]},
        {"id": "dup", "question": "  write   add(a, b) AND return the sum.  ", "tests": ["assert add(0, 0) == 0"]},
        {"id": "prefix", "question": FROZEN + "a different ending.", "tests": ["assert f() == 1"]},
        {"id": "grams", "question": ("unrelated opening. " * 30) + SHARED, "tests": ["assert f() == 1"]},
    ]
    kept, counts = apply_filters(rows, EXAM)
    assert [row["id"] for row in kept] == ["keep"], kept
    assert counts["empty_tests"] == 1
    assert counts["duplicate"] == 1
    assert counts["leak_prefix"] == 1
    assert counts["leak_ngram"] == 1
    assert leaks(rows[-1]["question"], EXAM) == "ngram"


def test_short_shared_phrase_is_not_five_grams():
    exam = [{"id": "e", "question": "please add two numbers safely"}]
    assert char_grams(exam[0]["question"]) == set()
    assert leaks("please add two numbers today", exam) is None
    assert leaks(exam[0]["question"], exam) == "prefix"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
