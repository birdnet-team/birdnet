import numpy as np
import pytest

from birdnet.acoustic.inference.configs import (
  ProcessingConfig,
)


def test_zero_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"max_audio_duration_min must be > 0",
  ):
    ProcessingConfig.validate_max_audio_duration_min(0)


def test_negative_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"max_audio_duration_min must be > 0",
  ):
    ProcessingConfig.validate_max_audio_duration_min(-5)


def test_non_number_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"max_audio_duration_min must be a number",
  ):
    ProcessingConfig.validate_max_audio_duration_min("long")  # type: ignore


def test_valid_integer_value() -> None:
  assert ProcessingConfig.validate_max_audio_duration_min(10) == 10


def test_valid_float_value() -> None:
  assert ProcessingConfig.validate_max_audio_duration_min(7.5) == 7.5


def test_valid_small_value() -> None:
  assert ProcessingConfig.validate_max_audio_duration_min(0.1) == 0.1


def test_valid_large_value() -> None:
  assert ProcessingConfig.validate_max_audio_duration_min(1e6) == 1e6


@pytest.mark.no_tf
def test_numpy_scalar_becomes_a_python_number() -> None:
  result = ProcessingConfig.validate_max_audio_duration_min(np.float32(1.5))
  assert result == 1.5
  assert type(result) is float
  assert type(ProcessingConfig.validate_max_audio_duration_min(2)) is int
