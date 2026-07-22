import pytest

from birdnet.acoustic.inference.configs import PredictionConfig


def test_float_is_valid() -> None:
  assert PredictionConfig.validate_default_confidence_threshold(0.5) == 0.5


def test_integer_is_valid() -> None:
  assert PredictionConfig.validate_default_confidence_threshold(1) == 1


def test_negative_is_returned_as_is() -> None:
  # The validator only enforces the type, not the range.
  assert PredictionConfig.validate_default_confidence_threshold(-0.5) == -0.5


def test_string_raises_error() -> None:
  with pytest.raises(
    TypeError, match=r"default confidence threshold must be a number"
  ):
    PredictionConfig.validate_default_confidence_threshold("high")  # type: ignore


def test_none_raises_error() -> None:
  with pytest.raises(
    TypeError, match=r"default confidence threshold must be a number"
  ):
    PredictionConfig.validate_default_confidence_threshold(None)  # type: ignore
