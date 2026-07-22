"""Validators of the geo inference configs (latitude/longitude/week/...)."""

import pytest

from birdnet.geo.inference.configs import (
  PredictionConfig,
  ProcessingConfig,
  RunConfig,
)

# ----------------------------------- latitude ------------------------------------


@pytest.mark.parametrize("value", [-90, 0, 42.5, 90])
def test_validate_latitude_valid(value: float) -> None:
  result = RunConfig.validate_latitude(value)
  assert result == float(value)
  assert isinstance(result, float)


@pytest.mark.parametrize("value", [-90.1, 90.1, 1000])
def test_validate_latitude_out_of_range_raises_error(value: float) -> None:
  with pytest.raises(ValueError, match=r"'latitude' is invalid"):
    RunConfig.validate_latitude(value)


def test_validate_latitude_non_number_raises_error() -> None:
  with pytest.raises(TypeError, match=r"latitude must be a float"):
    RunConfig.validate_latitude("42")  # type: ignore


# ----------------------------------- longitude -----------------------------------


@pytest.mark.parametrize("value", [-180, 0, -76.45, 180])
def test_validate_longitude_valid(value: float) -> None:
  result = RunConfig.validate_longitude(value)
  assert result == float(value)
  assert isinstance(result, float)


@pytest.mark.parametrize("value", [-180.1, 180.1])
def test_validate_longitude_out_of_range_raises_error(value: float) -> None:
  with pytest.raises(ValueError, match=r"'longitude' is invalid"):
    RunConfig.validate_longitude(value)


def test_validate_longitude_non_number_raises_error() -> None:
  with pytest.raises(TypeError, match=r"longitude must be a float"):
    RunConfig.validate_longitude("0")  # type: ignore


# ------------------------------------- week --------------------------------------


def test_validate_week_none_is_valid() -> None:
  assert RunConfig.validate_week(None) is None


@pytest.mark.parametrize("value", [1, 24, 48])
def test_validate_week_valid(value: int) -> None:
  assert RunConfig.validate_week(value) == value


@pytest.mark.parametrize("value", [0, 49])
def test_validate_week_out_of_range_raises_error(value: int) -> None:
  with pytest.raises(ValueError, match=r"'week' is invalid"):
    RunConfig.validate_week(value)


def test_validate_week_non_integer_raises_error() -> None:
  with pytest.raises(TypeError, match=r"'week' is invalid! It must be an integer"):
    RunConfig.validate_week(4.5)  # type: ignore


# ------------------------------ year_round_aggregation ---------------------------


@pytest.mark.parametrize("value", ["max", "average"])
def test_validate_year_round_aggregation_valid(value: str) -> None:
  assert RunConfig.validate_year_round_aggregation(value) == value


def test_validate_year_round_aggregation_invalid_raises_error() -> None:
  with pytest.raises(ValueError, match=r"'year_round_aggregation' is invalid"):
    RunConfig.validate_year_round_aggregation("median")  # type: ignore


# --------------------------------- min_confidence --------------------------------


@pytest.mark.parametrize("value", [0.0, 0.03, 0.999])
def test_validate_min_confidence_valid(value: float) -> None:
  result = PredictionConfig.validate_min_confidence(value)
  assert result == float(value)
  assert isinstance(result, float)


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_validate_min_confidence_out_of_range_raises_error(value: float) -> None:
  with pytest.raises(ValueError, match=r"'min_confidence' is invalid"):
    PredictionConfig.validate_min_confidence(value)


# ------------------------------------ device -------------------------------------


@pytest.mark.parametrize("value", ["CPU", "GPU", "GPU:0"])
def test_validate_device_valid(value: str) -> None:
  assert ProcessingConfig.validate_device(value) == value


def test_validate_device_invalid_raises_error() -> None:
  with pytest.raises(ValueError, match=r"device name must contain 'CPU' or 'GPU'"):
    ProcessingConfig.validate_device("TPU")


def test_validate_device_non_string_raises_error() -> None:
  with pytest.raises(TypeError, match=r"device must be a string"):
    ProcessingConfig.validate_device(42)  # type: ignore


# --------------------------------- half_precision --------------------------------


def test_validate_half_precision_valid() -> None:
  assert ProcessingConfig.validate_half_precision(True) is True
  assert ProcessingConfig.validate_half_precision(False) is False


def test_validate_half_precision_non_bool_raises_error() -> None:
  with pytest.raises(TypeError, match=r"half precision must be a boolean"):
    ProcessingConfig.validate_half_precision(1)  # type: ignore
