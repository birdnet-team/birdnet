import pytest

from birdnet.models.model_v2m4_tflite import ModelV2M4TFLite
from birdnet_tests.models.test_predict_species_at_location_and_time import (
  model_test_identical_predictions_return_same_result, model_test_no_week,
  model_test_using_no_threshold_returns_all_species, model_test_using_threshold)


@pytest.fixture(name="model")
def get_model():
  model = ModelV2M4TFLite(language="en_us")
  return model


def test_no_week(model: ModelV2M4TFLite):
  model_test_no_week(model)


def test_using_threshold(model: ModelV2M4TFLite):
  model_test_using_threshold(model)


def test_using_no_threshold_returns_all_species(model: ModelV2M4TFLite):
  model_test_using_no_threshold_returns_all_species(model)


def test_identical_predictions_return_same_result(model: ModelV2M4TFLite):
  model_test_identical_predictions_return_same_result(model)