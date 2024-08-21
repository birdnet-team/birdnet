import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest
from tqdm import tqdm

from birdnet.models.v2m4.model_v2m4_tflite import AudioModelV2M4TFLite
from birdnet.types import Species, SpeciesPredictions
from birdnet_tests.helper import TEST_RESULTS_DIR, species_predictions_are_equal
from birdnet_tests.models.m2v4.test_predict_species_within_audio_file import (
  TEST_FILE_WAV, AudioTestCase, create_ground_truth_test_file,
  model_minimum_test_soundscape_predictions_are_correct,
  model_test_identical_predictions_return_same_result,
  model_test_soundscape_predictions_are_globally_correct, predict_species_within_audio_file)

TEST_PATH = Path(TEST_RESULTS_DIR / "v2m4" / "audio-model.tflite.pkl")


@pytest.fixture(name="model")
def provide_model_to_tests():
  return get_model()


def get_model():
  model = AudioModelV2M4TFLite(language="en_us")
  return model


def test_soundscape_predictions_are_globally_correct(model: AudioModelV2M4TFLite):
  model_test_soundscape_predictions_are_globally_correct(model, precision=2)


def test_minimum_test_soundscape_predictions_are_correct(model: AudioModelV2M4TFLite):
  model_minimum_test_soundscape_predictions_are_correct(model, precision=2)


def test_identical_predictions_return_same_result(model: AudioModelV2M4TFLite):
  model_test_identical_predictions_return_same_result(model)


def test_invalid_species_filter_raises_value_error(model: AudioModelV2M4TFLite):
  invalid_filter_species: Set[Species] = {"species"}
  with pytest.raises(ValueError, match=rf"At least one species defined in 'filter_species' is invalid! They need to be known species, e.g., {', '.join(model.species[:3])}"):
    next(model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        filter_species=invalid_filter_species
    ))


def test_internal_predictions_are_correct(model: AudioModelV2M4TFLite):
  with TEST_PATH.open("rb") as f:
    test_cases: List[Tuple[Dict, SpeciesPredictions]] = pickle.load(f)

  for test_case_dict, gt in tqdm(test_cases):
    test_case = AudioTestCase(**test_case_dict)
    res = predict_species_within_audio_file(test_case,
                                            model, TEST_FILE_WAV)
    assert species_predictions_are_equal(res, gt, precision=7)


if __name__ == "__main__":
  m = get_model()
  create_ground_truth_test_file(m, TEST_PATH)