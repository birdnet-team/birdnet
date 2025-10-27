import multiprocessing

import psutil
import pytest

from birdnet.acoustic_models.inference_pipeline.configs import (
  ProcessingConfig,
)


def test_zero_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"workers must be >= 1",
  ):
    ProcessingConfig.validate_n_workers(0)


def test_negative_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"workers must be >= 1",
  ):
    ProcessingConfig.validate_n_workers(-1)


def test_non_integer_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"workers must be an integer",
  ):
    ProcessingConfig.validate_n_workers(1.5)  # type: ignore


def test_one_is_valid() -> None:
  assert ProcessingConfig.validate_n_workers(1) == 1


def test_max_cpus_is_valid() -> None:
  if max_logical_cpus := multiprocessing.cpu_count():
    assert ProcessingConfig.validate_n_workers(max_logical_cpus) == max_logical_cpus


def test_more_than_max_cpus_is_raises_error() -> None:
  if max_logical_cpus := multiprocessing.cpu_count():
    with pytest.raises(
      ValueError,
      match=rf"workers must be <= {max_logical_cpus}",
    ):
      ProcessingConfig.validate_n_workers(max_logical_cpus + 1)


def test_none_is_valid() -> None:
  n_physical_cores = psutil.cpu_count(logical=False) or 1
  assert ProcessingConfig.validate_n_workers(None) == n_physical_cores
