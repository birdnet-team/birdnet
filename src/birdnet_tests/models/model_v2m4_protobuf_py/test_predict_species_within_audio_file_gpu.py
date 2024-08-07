
from typing import Optional

import pytest
import tensorflow as tf

from birdnet.models.model_v2m4_protobuf import ModelV2M4Protobuf
from birdnet_tests.models.test_predict_species_within_audio_file import (
  model_minimum_test_soundscape_predictions_are_correct,
  model_test_identical_predictions_return_same_result,
  model_test_soundscape_predictions_are_correct)


@pytest.fixture(name="model")
def get_model():
  all_gpus = tf.config.list_logical_devices('GPU')
  if len(all_gpus) > 0:
    first_gpu: tf.config.LogicalDevice = all_gpus[0]
    model = ModelV2M4Protobuf(language="en_us", custom_device=first_gpu.name)
    return model
  return None


def test_soundscape_predictions_are_correct(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_test_soundscape_predictions_are_correct(model, precision=2)


def test_minimum_test_soundscape_predictions_are_correct(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_minimum_test_soundscape_predictions_are_correct(model, precision=2)


def test_identical_predictions_return_same_result(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    model_test_identical_predictions_return_same_result(model)
