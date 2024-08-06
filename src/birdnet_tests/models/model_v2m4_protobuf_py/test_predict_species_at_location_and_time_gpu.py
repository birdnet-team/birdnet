from typing import Optional

import pytest
import tensorflow as tf

from birdnet.models.model_v2m4_protobuf import ModelV2M4Protobuf
from birdnet_tests.models.test_predict_species_at_location_and_time import (
  model_test_identical_predictions_return_same_result, model_test_no_week,
  model_test_using_no_threshold_returns_all_species, model_test_using_threshold)


@pytest.fixture(name="model")
def get_model():
  all_gpus = tf.config.list_logical_devices('GPU')
  if len(all_gpus) > 0:
    first_gpu: tf.config.LogicalDevice = all_gpus[0]
    model = ModelV2M4Protobuf(language="en_us", custom_device=first_gpu.name)
    return model
  return None


def test_no_week(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_test_no_week(model)


def test_using_threshold(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_test_using_threshold(model)


def test_using_no_threshold_returns_all_species(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_test_using_no_threshold_returns_all_species(model)


def test_identical_predictions_return_same_result(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_test_identical_predictions_return_same_result(model)
