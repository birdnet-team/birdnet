import pytest

from birdnet.acoustic_models.inference_pipeline.configs import (
  ProcessingConfig,
)


def test_zero_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"batch size must be >= 1",
  ):
    ProcessingConfig.validate_batch_size(0)


def test_negative_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"batch size must be >= 1",
  ):
    ProcessingConfig.validate_batch_size(-1)


def test_one_is_valid() -> None:
  assert ProcessingConfig.validate_batch_size(1) == 1


def test_non_integer_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"batch size must be an integer",
  ):
    ProcessingConfig.validate_batch_size(1.5)  # type: ignore


def test_string_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"batch size must be an integer",
  ):
    ProcessingConfig.validate_batch_size("large")  # type: ignore


def test_large_value_is_valid() -> None:
  assert ProcessingConfig.validate_batch_size(1000) == 1000


def test_very_large_value_is_valid() -> None:
  assert ProcessingConfig.validate_batch_size(2**32) == 2**32
