import pytest

from birdnet.utils.segments import apply_speed_to_duration


def test_apply_speed_to_duration() -> None:
  assert apply_speed_to_duration(10.0, 2.0) == 20.0
  assert apply_speed_to_duration(10.0, 0.5) == 5.0


def test_apply_speed_to_duration_rejects_non_positive_speed() -> None:
  with pytest.raises(AssertionError):
    apply_speed_to_duration(10.0, 0.0)
