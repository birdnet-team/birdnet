import re

import numpy as np
import pytest

from birdnet.acoustic.inference.configs import (
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


def test_valid_small_value() -> None:
  assert ProcessingConfig.validate_overlap_duration(0.1, 3) == 0.1


def test_valid_large_value() -> None:
  assert ProcessingConfig.validate_overlap_duration(2.9, 3) == 2.9


@pytest.mark.no_tf
def test_numpy_scalar_becomes_a_python_number() -> None:
  result = ProcessingConfig.validate_overlap_duration(np.float32(0.5), 3.0)
  assert result == 0.5
  assert type(result) is float
  assert type(ProcessingConfig.validate_overlap_duration(np.int64(1), 3.0)) is int


@pytest.mark.no_tf
@pytest.mark.parametrize("value", [1, 0.5, np.float64(0.5), True])
def test_a_builtin_number_is_returned_as_it_was(value: float) -> None:
  # np.float64 and bool subclass float and int; both always passed unchanged.
  assert ProcessingConfig.validate_overlap_duration(value, 3.0) is value
