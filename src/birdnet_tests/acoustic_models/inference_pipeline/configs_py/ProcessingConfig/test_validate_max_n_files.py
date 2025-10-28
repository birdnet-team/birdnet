import pytest

from birdnet.acoustic_models.inference_pipeline.configs import ProcessingConfig


def test_0_raise_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"max_n_files must be >= 1",
  ):
    ProcessingConfig.validate_max_n_files(0)


def test_negative_raise_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"max_n_files must be >= 1",
  ):
    ProcessingConfig.validate_max_n_files(-5)


def test_too_large_raise_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"max_n_files must be <= 2\^64",
  ):
    ProcessingConfig.validate_max_n_files(2**64 + 1)


def test_non_integer_raise_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"max_n_files must be an integer",
  ):
    ProcessingConfig.validate_max_n_files(3.14)  # type: ignore


def test_valid_value() -> None:
  assert ProcessingConfig.validate_max_n_files(100) == 100


def test_valid_max_value() -> None:
  assert ProcessingConfig.validate_max_n_files(2**64) == 2**64


def test_valid_min_value() -> None:
  assert ProcessingConfig.validate_max_n_files(1) == 1
