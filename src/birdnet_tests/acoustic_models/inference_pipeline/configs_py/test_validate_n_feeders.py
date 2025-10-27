import multiprocessing

import pytest

from birdnet.acoustic_models.inference_pipeline.configs import (
  ProcessingConfig,
)


def test_zero_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"feeders must be >= 1",
  ):
    ProcessingConfig.validate_n_feeders(0)


def test_negative_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"feeders must be >= 1",
  ):
    ProcessingConfig.validate_n_feeders(-1)


def test_non_integer_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"feeders must be an integer",
  ):
    ProcessingConfig.validate_n_feeders(1.5)  # type: ignore


def test_one_is_valid() -> None:
  assert ProcessingConfig.validate_n_feeders(1) == 1


def test_max_cpus_is_valid() -> None:
  if max_logical_cpus := multiprocessing.cpu_count():
    assert ProcessingConfig.validate_n_feeders(max_logical_cpus) == max_logical_cpus


def test_more_than_max_cpus_is_raises_error() -> None:
  if max_logical_cpus := multiprocessing.cpu_count():
    with pytest.raises(
      ValueError,
      match=rf"feeders must be <= {max_logical_cpus}",
    ):
      ProcessingConfig.validate_n_feeders(max_logical_cpus + 1)

