from typing import Set

import pytest

from birdnet.models.model_v2m4_tflite import ModelV2M4TFLite
from birdnet.types import Species
from birdnet_tests.models.test_predict_species_within_audio_file import (
  TEST_FILE_WAV, model_minimum_test_soundscape_predictions_are_correct,
  model_test_identical_predictions_return_same_result,
  model_test_soundscape_predictions_are_correct)


@pytest.fixture(name="model")
def get_model():
  model = ModelV2M4TFLite(language="en_us")
  return model


def test_soundscape_predictions_are_correct(model: ModelV2M4TFLite):
  model_test_soundscape_predictions_are_correct(model, precision=2)


def test_minimum_test_soundscape_predictions_are_correct(model: ModelV2M4TFLite):
  model_minimum_test_soundscape_predictions_are_correct(model, precision=2)


def test_identical_predictions_return_same_result(model: ModelV2M4TFLite):
  model_test_identical_predictions_return_same_result(model)


def test_invalid_species_filter_raises_value_error(model: ModelV2M4TFLite):
  invalid_filter_species: Set[Species] = {"species"}
  with pytest.raises(ValueError, match=rf"At least one species defined in 'filter_species' is invalid! They need to be known species, e.g., {', '.join(model._species_list[:3])}"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        filter_species=invalid_filter_species
    )
