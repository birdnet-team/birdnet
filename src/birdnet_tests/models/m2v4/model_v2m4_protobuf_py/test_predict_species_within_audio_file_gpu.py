import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import tensorflow as tf
from tqdm import tqdm

from birdnet.models.v2m4.model_v2m4_protobuf import AudioModelV2M4Protobuf
from birdnet.types import SpeciesPredictions
from birdnet_tests.helper import TEST_RESULTS_DIR, species_predictions_are_equal
from birdnet_tests.models.m2v4.test_predict_species_within_audio_file import (
  TEST_FILE_WAV, AudioTestCase, create_ground_truth_test_file,
  model_minimum_test_soundscape_predictions_are_correct,
  model_test_identical_predictions_return_same_result,
  model_test_soundscape_predictions_are_globally_correct,
  predict_species_within_audio_file_in_test_case)

TEST_PATH = Path(TEST_RESULTS_DIR / "v2m4" / "audio-model.protobuf-gpu.pkl")


@pytest.fixture(name="model")
def provide_model_to_tests():
  return get_model()


def get_model():
  all_gpus = tf.config.list_logical_devices('GPU')
  if len(all_gpus) > 0:
    first_gpu: tf.config.LogicalDevice = all_gpus[0]
    model = AudioModelV2M4Protobuf(language="en_us", custom_device=first_gpu.name)
    return model
  return None


def test_soundscape_predictions_are_globally_correct(model: Optional[AudioModelV2M4Protobuf]):
  if model is not None:
    model_test_soundscape_predictions_are_globally_correct(model, precision=2)


def test_minimum_test_soundscape_predictions_are_correct(model: Optional[AudioModelV2M4Protobuf]):
  if model is not None:
    model_minimum_test_soundscape_predictions_are_correct(model, precision=2)


def test_identical_predictions_return_same_result(model: Optional[AudioModelV2M4Protobuf]):
  if model is not None:
    model_test_identical_predictions_return_same_result(model)


def test_internal_predictions_are_correct(model: Optional[AudioModelV2M4Protobuf]):
  if model is not None:
    with TEST_PATH.open("rb") as f:
      test_cases: List[Tuple[Dict, SpeciesPredictions]] = pickle.load(f)

    for test_case_dict, gt in tqdm(test_cases):
      test_case = AudioTestCase(**test_case_dict)
      res = predict_species_within_audio_file_in_test_case(test_case,
                                                           model, TEST_FILE_WAV)
      # has some variations
      assert species_predictions_are_equal(res, gt, decimal=4)


if __name__ == "__main__":
  m = get_model()
  assert m is not None
  create_ground_truth_test_file(m, TEST_PATH)
