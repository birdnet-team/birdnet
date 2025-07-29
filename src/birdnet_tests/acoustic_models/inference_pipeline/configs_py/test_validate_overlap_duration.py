import re

import pytest

from birdnet.acoustic_models.inference_pipeline.configs import (
  ProcessingConfig,
)


def test_negative_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=re.escape("overlap duration must be in [0, 3)"),
  ):
    ProcessingConfig.validate_overlap_duration(-1, 3)


def test_same_as_seg_size_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=re.escape("overlap duration must be in [0, 3)"),
  ):
    ProcessingConfig.validate_overlap_duration(3, 3)


def test_larger_than_seg_size_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=re.escape("overlap duration must be in [0, 3)"),
  ):
    ProcessingConfig.validate_overlap_duration(4, 3)


def test_zero_is_valid() -> None:
  assert ProcessingConfig.validate_overlap_duration(0, 3) == 0


def test_one_is_valid() -> None:
  assert ProcessingConfig.validate_overlap_duration(1, 3) == 1


def test_float_is_valid() -> None:
  assert ProcessingConfig.validate_overlap_duration(1.5, 3) == 1.5


def test_non_number_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"overlap duration must be a number",
  ):
    ProcessingConfig.validate_overlap_duration("1", 3)  # type: ignore
