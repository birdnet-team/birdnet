import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from tqdm import tqdm

from birdnet.models.v2m4.model_v2m4_tflite import MetaModelV2M4TFLite
from birdnet.types import SpeciesPrediction
from birdnet_tests.helper import TEST_RESULTS_DIR, species_prediction_is_equal
from birdnet_tests.models.m2v4.test_predict_species_at_location_and_time import (
  LocationTestCase, create_ground_truth_test_file,
  model_test_identical_predictions_return_same_result, model_test_no_week,
  model_test_predictions_are_globally_correct, model_test_using_no_threshold_returns_all_species,
  model_test_using_threshold, predict_species)

TEST_PATH = Path(TEST_RESULTS_DIR / "v2m4" / "meta-model.tflite.pkl")


@pytest.fixture(name="model")
def provide_model_to_tests():
  return get_model()


def get_model():
  model = MetaModelV2M4TFLite(language="en_us")
  return model


def test_predictions_are_globally_correct(model: MetaModelV2M4TFLite):
  model_test_predictions_are_globally_correct(model, precision=5)


def test_no_week(model: MetaModelV2M4TFLite):
  model_test_no_week(model)


def test_using_threshold(model: MetaModelV2M4TFLite):
  model_test_using_threshold(model)


def test_using_no_threshold_returns_all_species(model: MetaModelV2M4TFLite):
  model_test_using_no_threshold_returns_all_species(model)


def test_identical_predictions_return_same_result(model: MetaModelV2M4TFLite):
  model_test_identical_predictions_return_same_result(model)


def test_internal_predictions_are_correct(model: MetaModelV2M4TFLite):
  with TEST_PATH.open("rb") as f:
    test_cases: List[Tuple[Dict, SpeciesPrediction]] = pickle.load(f)

  for test_case_dict, gt in tqdm(test_cases):
    test_case = LocationTestCase(**test_case_dict)
    res = predict_species(test_case, model)
    assert species_prediction_is_equal(res, gt, decimal=5)


if __name__ == "__main__":
  m = get_model()
  create_ground_truth_test_file(m, TEST_PATH)
