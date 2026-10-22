"""Unit tests for GSPO group shaping. No GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import dapo_keep, length_penalize, shape_group


def test_dapo_drops_all_fail_and_all_pass():
    assert dapo_keep(0.0) is False
    assert dapo_keep(1.0) is False


def test_dapo_keeps_open_interval():
    assert dapo_keep(0.25) is True
    assert dapo_keep(0.8) is True
    assert dapo_keep(0.81) is False


def test_length_shortest_correct_is_one_longer_is_less_wrong_stays_zero():
    out = length_penalize([0.0, 1.0, 1.0, 0.0], [400, 100, 400, 200], max_len=512)
    assert out[0] == 0.0 and out[3] == 0.0
    assert out[1] == 1.0
    assert 0.0 < out[2] < 1.0


def test_length_no_correct_stays_zero():
    assert length_penalize([0.0, 0.0], [10, 99], max_len=512) == [0.0, 0.0]


def test_shape_group_zeros_when_dapo_drops():
    # all correct → mean 1 after binary, drop, advantages would be 0
    shaped = shape_group([1.0, 1.0, 1.0, 1.0], [80, 90, 100, 110], max_len=512)
    assert shaped == [0.0, 0.0, 0.0, 0.0]


def test_shape_group_keeps_mixed_and_penalizes_long_correct():
    shaped = shape_group([0.0, 1.0, 1.0, 0.0], [50, 80, 200, 60], max_len=512)
    assert shaped[0] == 0.0 and shaped[3] == 0.0
    assert shaped[1] == 1.0
    assert 0.0 < shaped[2] < 1.0


if __name__ == "__main__":
    test_dapo_drops_all_fail_and_all_pass()
    test_dapo_keeps_open_interval()
    test_length_shortest_correct_is_one_longer_is_less_wrong_stays_zero()
    test_length_no_correct_stays_zero()
    test_shape_group_zeros_when_dapo_drops()
    test_shape_group_keeps_mixed_and_penalizes_long_correct()
    print("ok")
