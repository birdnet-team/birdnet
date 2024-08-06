
import pytest
import tensorflow as tf

from birdnet.models.model_v2m4_protobuf import ModelV2M4Protobuf
from birdnet_tests.models.test_predict_species_within_audio_file import (
  model_test_identical_predictions_return_same_result,
  model_test_soundscape_predictions_are_correct,
  model_test_soundscape_predictions_batch_size_4_are_correct)


@pytest.fixture(name="model")
def get_model():
  all_gpus = tf.config.list_logical_devices('GPU')
  if len(all_gpus) > 0:
    first_gpu: tf.config.LogicalDevice = all_gpus[0]
    model = ModelV2M4Protobuf(language="en_us", custom_device=first_gpu.name)
    return model
  return None


def test_soundscape_predictions_are_correct(model: ModelV2M4Protobuf):
  model_test_soundscape_predictions_are_correct(model)


def test_soundscape_predictions_batch_size_4_are_correct(model: ModelV2M4Protobuf):
  model_test_soundscape_predictions_batch_size_4_are_correct(model)


def test_identical_predictions_return_same_result(model: ModelV2M4Protobuf):
  model_test_identical_predictions_return_same_result(model)
