import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy.testing as npt
import pytest
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.location_based_prediction import predict_species_at_location_and_time
from birdnet.models.v2m4.model_v2m4_base import MetaModelBaseV2M4
from birdnet.models.v2m4.model_v2m4_protobuf import MetaModelV2M4Protobuf
from birdnet.types import SpeciesPrediction
from birdnet_tests.helper import TEST_RESULTS_DIR, species_prediction_is_equal

TEST_PATH = Path(TEST_RESULTS_DIR / "v2m4" / "meta-model.pkl")


class DummyModel(MetaModelBaseV2M4):
  pass


@pytest.fixture(name="model")
def get_model():
  model = DummyModel(OrderedSet(("species1", "species2")))
  return model


def test_invalid_latitude_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'latitude' is invalid! It needs to be in interval \[-90.0, 90.0\]."):
    predict_species_at_location_and_time(91.0, 0, custom_model=model)

  with pytest.raises(ValueError, match=r"Value for 'latitude' is invalid! It needs to be in interval \[-90.0, 90.0\]."):
    predict_species_at_location_and_time(-91.0, 0, custom_model=model)


def test_invalid_longitude_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'longitude' is invalid! It needs to be in interval \[-180.0, 180.0\]."):
    predict_species_at_location_and_time(0, 181.0, custom_model=model)

  with pytest.raises(ValueError, match=r"Value for 'longitude' is invalid! It needs to be in interval \[-180.0, 180.0\]."):
    predict_species_at_location_and_time(0, -181.0, custom_model=model)


def test_invalid_min_confidence_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    predict_species_at_location_and_time(0, 0, min_confidence=-0.1, custom_model=model)

  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    predict_species_at_location_and_time(0, 0, min_confidence=1.1, custom_model=model)


def test_invalid_week_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'week' is invalid! It needs to be either None or in interval \[1, 48\]."):
    predict_species_at_location_and_time(0, 0, week=0, custom_model=model)

  with pytest.raises(ValueError, match=r"Value for 'week' is invalid! It needs to be either None or in interval \[1, 48\]."):
    predict_species_at_location_and_time(0, 0, week=49, custom_model=model)


def model_test_no_week(model: MetaModelBaseV2M4):
  species = predict_species_at_location_and_time(
    42.5, -76.45, min_confidence=0.03, custom_model=model)
  assert len(species) >= 252  # 255 on TFLite

  assert list(species.keys())[0] == 'Cyanocitta cristata_Blue Jay'
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9586712, decimal=3)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.06815465, decimal=3)


def model_test_using_threshold(model: MetaModelBaseV2M4):
  species = predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0.03, custom_model=model)
  assert len(species) == 64
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9276199, decimal=3)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.035001162, decimal=3)
  assert list(species.keys())[0] == 'Cyanocitta cristata_Blue Jay'
  # is last or second last
  assert 'Larus marinus_Great Black-backed Gull' in list(species.keys())[-2:]


def model_test_using_no_threshold_returns_all_species(model: MetaModelBaseV2M4):
  species = predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0, custom_model=model)
  assert len(species) == len(model.species) == 6522


def model_test_identical_predictions_return_same_result(model: MetaModelBaseV2M4):
  species1 = predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0, custom_model=model)

  species2 = predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0, custom_model=model)

  assert species_prediction_is_equal(species1, species2, decimal=5)


@dataclass()
class LocationTestCase():
  latitude: float = 42.5
  longitude: float = -76.45
  week: Optional[int] = None


def predict_species(test_case: LocationTestCase, model: MetaModelBaseV2M4) -> SpeciesPrediction:
  # min_confidence=0 because otherwise the length is not always the same
  return predict_species_at_location_and_time(
    test_case.latitude, test_case.longitude, week=test_case.week, min_confidence=0, custom_model=model
  )


def create_ground_truth_test_file(model: MetaModelBaseV2M4, path: Path):
  # Ground truth is created using Protobuf CPU model

  test_cases = [
    LocationTestCase(),
    LocationTestCase(longitude=180),
    LocationTestCase(latitude=90),
    LocationTestCase(longitude=-180),
    LocationTestCase(latitude=-90),
    LocationTestCase(latitude=-90, longitude=180),
    LocationTestCase(week=22),
    LocationTestCase(latitude=-90, longitude=180, week=22),
  ]

  results = []
  for test_case in tqdm(test_cases):
    gt = predict_species(test_case, model)
    n_predictions = sum(1 for x in gt.values() if x > 0)
    assert n_predictions >= 5
    test_case_dict = asdict(test_case)
    results.append((test_case_dict, gt))

  with path.open("wb") as f:
    pickle.dump(results, f)


def model_test_predictions_are_globally_correct(model: MetaModelBaseV2M4, /, *, precision: int):
  with TEST_PATH.open("rb") as f:
    test_cases: List[Tuple[Dict, SpeciesPrediction]] = pickle.load(f)

  for test_case_dict, gt in tqdm(test_cases):
    test_case = LocationTestCase(**test_case_dict)
    res = predict_species(test_case, model)
    assert species_prediction_is_equal(res, gt, decimal=precision)


if __name__ == "__main__":
  # global ground truth is created using protobuf CPU model
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  m = MetaModelV2M4Protobuf(language="en_us", custom_device="/device:CPU:0")
  create_ground_truth_test_file(m, TEST_PATH)
