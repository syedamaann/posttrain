"""Unit tests for GSPO group shaping. No GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import (
    closed_verify, dapo_keep, nothink_verify, overlong_penalize, shape_group, think_answer,
)


def test_dapo_drops_all_fail_and_all_pass():
    assert dapo_keep(0.0) is False
    assert dapo_keep(1.0) is False


def test_dapo_keeps_open_interval():
    assert dapo_keep(0.25) is True
    assert dapo_keep(0.8) is True
    assert dapo_keep(0.81) is False


def test_overlong_full_credit_then_linear_to_zero_at_the_cap():
    # Window opens at 1024-256=768. 454 and 700 tie. 813 decays. Wrong stays 0.
    out = overlong_penalize([1.0, 1.0, 1.0, 0.0], [454, 700, 813, 100], max_len=1024)
    assert out[0] == 1.0 and out[1] == 1.0 and out[3] == 0.0
    assert abs(out[2] - (1024 - 813) / 256) < 1e-9


def test_overlong_no_correct_stays_zero():
    assert overlong_penalize([0.0, 0.0], [10, 900], max_len=1024) == [0.0, 0.0]


def test_shape_group_zeros_when_dapo_drops():
    # all correct → mean 1 after binary, drop, advantages would be 0
    shaped = shape_group([1.0, 1.0, 1.0, 1.0], [80, 90, 100, 110], max_len=512)
    assert shaped == [0.0, 0.0, 0.0, 0.0]


def test_shape_group_keeps_mixed_and_decays_only_near_the_cap():
    shaped = shape_group([0.0, 1.0, 1.0, 0.0], [50, 454, 900, 60], max_len=1024)
    assert shaped[0] == 0.0 and shaped[3] == 0.0
    assert shaped[1] == 1.0
    assert abs(shaped[2] - (1024 - 900) / 256) < 1e-9


def test_shape_group_drops_seven_of_eight_correct():
    binary = [1.0] * 7 + [0.0]
    assert shape_group(binary, [400] * 8, max_len=1024) == [0.0] * 8


def test_unclosed_text_has_no_answer_span():
    assert think_answer("the answer is \\boxed{18}") is None


def test_box_before_close_is_outside_the_scored_span():
    text = "<think>\\boxed{18}</think>\nrestated, no box"
    span = think_answer(text)
    assert span is not None
    assert "\\boxed{18}" not in span


def test_unclosed_box_scores_zero_even_when_verify_would_pass():
    assert closed_verify("\\boxed{18}", "18", lambda gold, span: True, think_tokens=400) == 0.0


def test_instant_close_scores_zero():
    assert closed_verify("</think>\n\\boxed{18}", "18", lambda gold, span: True, think_tokens=24) == 0.0


def test_think_body_floor_is_128_tokens():
    text = "<think>work</think>\n\\boxed{18}"
    verify = lambda gold, span: True
    assert closed_verify(text, "18", verify, think_tokens=127) == 0.0
    assert closed_verify(text, "18", verify, think_tokens=128) == 1.0


def test_bare_number_after_close_scores_zero():
    text = "<think>work</think>\n18"
    assert closed_verify(text, "18", lambda gold, span: True, think_tokens=400) == 0.0


def test_janet_group_pays_a_finished_proof():
    """S1, GSM8K test[0]. The 399-token unclosed box must not outrank a real proof."""
    def verify(gold, span):
        return "\\boxed{18}" in span

    rows = [
        (869, "<think>work</think>\n\\boxed{2}"),
        (511, "<think>work</think>\n\\boxed{18}"),
        (473, "<think>work</think>\n\\boxed{18}"),
        (813, "<think>work</think>\n\\boxed{18}"),
        (454, "<think>work</think>\n\\boxed{18}"),
        (399, "\\boxed{18}"),
        (1024, "<think>\\boxed{18}</think>\ncut before the answer"),
        (786, "<think>work</think>\n\\boxed{18}"),
    ]
    binary = [closed_verify(text, "18", verify, think_tokens=n) for n, text in rows]
    shaped = shape_group(binary, [n for n, _ in rows], max_len=1024)
    assert binary == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert shaped[5] == 0.0 and shaped[6] == 0.0 and shaped[0] == 0.0
    assert shaped[4] == 1.0 and shaped[2] == 1.0 and shaped[1] == 1.0
    assert abs(shaped[3] - (1024 - 813) / 256) < 1e-9
    assert abs(shaped[7] - (1024 - 786) / 256) < 1e-9


def test_nothink_duplicate_with_close_tag_scores_zero():
    """checkpoint-1000 item 0: answer, stray </think>, same answer again."""
    text = "Janet sells 9 eggs.\n\\boxed{18}\n</think>\nJanet sells 9 eggs.\n\\boxed{18}"
    assert nothink_verify(text, "18", lambda gold, span: True) == 0.0


def test_nothink_open_tag_scores_zero():
    text = "<think>\nJanet sells 9 eggs.\n\\boxed{18}"
    assert nothink_verify(text, "18", lambda gold, span: True) == 0.0


def test_nothink_clean_box_scores_one():
    text = "Janet sells 9 eggs.\n\\boxed{18}"
    assert nothink_verify(text, "18", lambda gold, span: True) == 1.0


def test_nothink_wrong_box_scores_zero():
    text = "Janet sells 9 eggs.\n\\boxed{2}"
    assert nothink_verify(text, "18", lambda gold, span: False) == 0.0


def test_nothink_bare_number_scores_zero():
    assert nothink_verify("18", "18", lambda gold, span: True) == 0.0


def test_nothink_group_pays_the_single_copy():
    """Five clean proofs and three duplicated ones. Only the clean copies are paid."""
    clean = "Janet sells 9 eggs.\n\\boxed{18}"
    leaked = "Janet sells 9 eggs.\n\\boxed{18}\n</think>\nJanet sells 9 eggs.\n\\boxed{18}"
    rows = [(80, clean), (146, clean), (196, clean), (117, clean), (279, clean), (259, leaked), (462, leaked), (379, leaked)]
    verify = lambda gold, span: True
    binary = [nothink_verify(text, "18", verify) for _, text in rows]
    shaped = shape_group(binary, [n for n, _ in rows], max_len=1024)
    assert binary == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert shaped[:5] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert shaped[5:] == [0.0, 0.0, 0.0]


if __name__ == "__main__":
    test_dapo_drops_all_fail_and_all_pass()
    test_dapo_keeps_open_interval()
    test_overlong_full_credit_then_linear_to_zero_at_the_cap()
    test_overlong_no_correct_stays_zero()
    test_shape_group_zeros_when_dapo_drops()
    test_shape_group_keeps_mixed_and_decays_only_near_the_cap()
    test_shape_group_drops_seven_of_eight_correct()
    test_unclosed_text_has_no_answer_span()
    test_box_before_close_is_outside_the_scored_span()
    test_unclosed_box_scores_zero_even_when_verify_would_pass()
    test_instant_close_scores_zero()
    test_think_body_floor_is_128_tokens()
    test_bare_number_after_close_scores_zero()
    test_janet_group_pays_a_finished_proof()
    test_nothink_duplicate_with_close_tag_scores_zero()
    test_nothink_open_tag_scores_zero()
    test_nothink_clean_box_scores_one()
    test_nothink_wrong_box_scores_zero()
    test_nothink_bare_number_scores_zero()
    test_nothink_group_pays_the_single_copy()
    print("ok")
