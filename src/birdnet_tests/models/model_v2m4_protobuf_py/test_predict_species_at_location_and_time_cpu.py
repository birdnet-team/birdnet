import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from tqdm import tqdm

from birdnet.models.model_v2m4_protobuf import ModelV2M4Protobuf
from birdnet.types import SpeciesPrediction
from birdnet_tests.helper import TEST_FILES_DIR, species_prediction_is_equal
from birdnet_tests.models.test_predict_species_at_location_and_time import (
  LocationTestCase, create_ground_truth_test_file,
  model_test_identical_predictions_return_same_result, model_test_no_week,
  model_test_predictions_are_globally_correct, model_test_using_no_threshold_returns_all_species,
  model_test_using_threshold, predict_species)


@pytest.fixture(name="model")
def get_model():
  model = ModelV2M4Protobuf(language="en_us", custom_device="/device:CPU:0")
  return model


def test_no_week(model: ModelV2M4Protobuf):
  model_test_no_week(model)


def test_using_threshold(model: ModelV2M4Protobuf):
  model_test_using_threshold(model)


def test_using_no_threshold_returns_all_species(model: ModelV2M4Protobuf):
  model_test_using_no_threshold_returns_all_species(model)


def test_identical_predictions_return_same_result(model: ModelV2M4Protobuf):
  model_test_identical_predictions_return_same_result(model)


def test_predictions_are_globally_correct(model: ModelV2M4Protobuf):
  model_test_predictions_are_globally_correct(model, precision=7)


TEST_PATH = Path(TEST_FILES_DIR / "meta-model.protobuf-cpu.pkl")


def test_internal_predictions_are_correct(model: Optional[ModelV2M4Protobuf]):
  if model is not None:
    with TEST_PATH.open("rb") as f:
      test_cases: List[Tuple[Dict, SpeciesPrediction]] = pickle.load(f)

    for test_case_dict, gt in tqdm(test_cases):
      test_case = LocationTestCase(**test_case_dict)
      res = predict_species(test_case, model)
      assert species_prediction_is_equal(res, gt, precision=7)


if __name__ == "__main__":
  model = ModelV2M4Protobuf(language="en_us", custom_device="/device:CPU:0")
  create_ground_truth_test_file(model, TEST_PATH)
