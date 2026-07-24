import pytest

from birdnet.acoustic.inference.configs import PredictionConfig


def test_valid_value() -> None:
  assert PredictionConfig.validate_top_k(3, max_value=5) == 3


def test_lower_bound_is_valid() -> None:
  assert PredictionConfig.validate_top_k(1, max_value=5) == 1


def test_upper_bound_is_valid() -> None:
  assert PredictionConfig.validate_top_k(5, max_value=5) == 5


def test_zero_raises_error() -> None:
  with pytest.raises(ValueError, match=r"top k must be in the range \[1, 5\]"):
    PredictionConfig.validate_top_k(0, max_value=5)


def test_above_max_raises_error() -> None:
  with pytest.raises(ValueError, match=r"top k must be in the range \[1, 5\]"):
    PredictionConfig.validate_top_k(6, max_value=5)


def test_non_integer_raises_error() -> None:
  with pytest.raises(TypeError, match=r"top k must be an integer"):
    PredictionConfig.validate_top_k(1.5, max_value=5)  # type: ignore
