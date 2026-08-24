import pytest

from birdnet.acoustic.inference.configs import PredictionConfig

MODEL_SPECIES = ["species_a", "species_b", "species_c"]


def test_valid_thresholds() -> None:
  thresholds = {"species_a": 0.5, "species_b": 0.1}
  assert (
    PredictionConfig.validate_custom_confidence_thresholds(thresholds, MODEL_SPECIES)
    == thresholds
  )


def test_empty_dict_is_valid() -> None:
  assert PredictionConfig.validate_custom_confidence_thresholds({}, MODEL_SPECIES) == {}


def test_integer_value_is_valid() -> None:
  thresholds = {"species_a": 1}
  assert (
    PredictionConfig.validate_custom_confidence_thresholds(thresholds, MODEL_SPECIES)
    == thresholds
  )


def test_non_dict_raises_error() -> None:
  with pytest.raises(
    TypeError, match=r"custom confidence thresholds must be a dictionary"
  ):
    PredictionConfig.validate_custom_confidence_thresholds(
      [("species_a", 0.5)], MODEL_SPECIES  # type: ignore
    )


def test_non_string_key_raises_error() -> None:
  with pytest.raises(
    TypeError, match=r"custom confidence threshold keys must be strings"
  ):
    PredictionConfig.validate_custom_confidence_thresholds({1: 0.5}, MODEL_SPECIES)


def test_unknown_species_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"species 'unknown' is not available in the model"
  ):
    PredictionConfig.validate_custom_confidence_thresholds(
      {"unknown": 0.5}, MODEL_SPECIES
    )


def test_non_number_value_raises_error() -> None:
  with pytest.raises(
    TypeError, match=r"custom confidence threshold values must be numbers"
  ):
    PredictionConfig.validate_custom_confidence_thresholds(
      {"species_a": "high"}, MODEL_SPECIES
    )
