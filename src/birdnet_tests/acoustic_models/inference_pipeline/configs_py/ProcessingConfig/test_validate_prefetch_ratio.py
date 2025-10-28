import pytest

from birdnet.acoustic_models.inference_pipeline.configs import (
  ProcessingConfig,
)


def test_negative_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"prefetch_ratio must be >= 0",
  ):
    ProcessingConfig.validate_prefetch_ratio(-1)


def test_zero_is_valid() -> None:
  assert ProcessingConfig.validate_prefetch_ratio(0) == 0


def test_one_is_valid() -> None:
  assert ProcessingConfig.validate_prefetch_ratio(1) == 1


def test_non_integer_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"prefetch_ratio must be an integer",
  ):
    ProcessingConfig.validate_prefetch_ratio(1.5)  # type: ignore
