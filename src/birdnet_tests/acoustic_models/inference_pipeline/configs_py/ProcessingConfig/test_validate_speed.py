import re

import pytest

from birdnet.acoustic.inference.configs import ProcessingConfig


def test_one_is_valid() -> None:
  assert ProcessingConfig.validate_speed(1.0) == 1.0


def test_integer_is_coerced_to_float() -> None:
  result = ProcessingConfig.validate_speed(2)
  assert result == 2.0
  assert isinstance(result, float)


def test_lower_bound_is_valid() -> None:
  assert ProcessingConfig.validate_speed(0.01) == 0.01


def test_upper_bound_is_valid() -> None:
  assert ProcessingConfig.validate_speed(100.0) == 100.0


def test_too_small_raises_error() -> None:
  with pytest.raises(
    ValueError, match=re.escape("speed must be in the range [0.01, 100.0]")
  ):
    ProcessingConfig.validate_speed(0.005)


def test_too_large_raises_error() -> None:
  with pytest.raises(
    ValueError, match=re.escape("speed must be in the range [0.01, 100.0]")
  ):
    ProcessingConfig.validate_speed(100.1)


def test_non_number_raises_error() -> None:
  with pytest.raises(TypeError, match=r"speed must be a number"):
    ProcessingConfig.validate_speed("fast")  # type: ignore
