import numpy as np
import pytest

from birdnet.acoustic.inference.configs import PredictionConfig


def test_one_is_valid() -> None:
  assert PredictionConfig.validate_sigmoid_sensitivity(1.0) == 1.0


def test_lower_bound_is_valid() -> None:
  assert PredictionConfig.validate_sigmoid_sensitivity(0.5) == 0.5


def test_upper_bound_is_valid() -> None:
  assert PredictionConfig.validate_sigmoid_sensitivity(1.5) == 1.5


def test_too_small_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"sigmoid sensitivity must be in the range \[0.5, 1.5\]"
  ):
    PredictionConfig.validate_sigmoid_sensitivity(0.1)


def test_too_large_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"sigmoid sensitivity must be in the range \[0.5, 1.5\]"
  ):
    PredictionConfig.validate_sigmoid_sensitivity(2.0)


@pytest.mark.no_tf
def test_none_is_the_plain_sigmoid() -> None:
  assert PredictionConfig.validate_sigmoid_sensitivity(None) == 1.0


def test_non_number_raises_error() -> None:
  with pytest.raises(TypeError, match=r"sigmoid sensitivity must be a number"):
    PredictionConfig.validate_sigmoid_sensitivity("high")  # type: ignore


@pytest.mark.no_tf
def test_numpy_scalar_becomes_a_python_number() -> None:
  result = PredictionConfig.validate_sigmoid_sensitivity(np.float32(1.25))
  assert result == 1.25
  assert type(result) is float
  assert type(PredictionConfig.validate_sigmoid_sensitivity(1)) is int
