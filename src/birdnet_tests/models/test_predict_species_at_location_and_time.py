import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy.testing as npt
import pytest
from tqdm import tqdm

from birdnet.models.model_v2m4_base import ModelV2M4Base
from birdnet.models.model_v2m4_protobuf import ModelV2M4Protobuf
from birdnet.types import SpeciesPrediction
from birdnet_tests.helper import species_prediction_is_equal

TEST_FILES_DIR = Path("src/birdnet_tests/test_files")

TEST_PATH = Path(TEST_FILES_DIR / "meta-model.global.pkl")


class DummyModel(ModelV2M4Base):
  pass


@pytest.fixture(name="model")
def get_model():
  model = DummyModel(language="en_us")
  return model


def test_invalid_latitude_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'latitude' is invalid! It needs to be in interval \[-90.0, 90.0\]."):
    model.predict_species_at_location_and_time(91.0, 0)

  with pytest.raises(ValueError, match=r"Value for 'latitude' is invalid! It needs to be in interval \[-90.0, 90.0\]."):
    model.predict_species_at_location_and_time(-91.0, 0)


def test_invalid_longitude_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'longitude' is invalid! It needs to be in interval \[-180.0, 180.0\]."):
    model.predict_species_at_location_and_time(0, 181.0)

  with pytest.raises(ValueError, match=r"Value for 'longitude' is invalid! It needs to be in interval \[-180.0, 180.0\]."):
    model.predict_species_at_location_and_time(0, -181.0)


def test_invalid_min_confidence_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    model.predict_species_at_location_and_time(0, 0, min_confidence=-0.1)

  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    model.predict_species_at_location_and_time(0, 0, min_confidence=1.1)


def test_invalid_week_raises_value_error(model: DummyModel):
  with pytest.raises(ValueError, match=r"Value for 'week' is invalid! It needs to be either None or in interval \[1, 48\]."):
    model.predict_species_at_location_and_time(0, 0, week=0)

  with pytest.raises(ValueError, match=r"Value for 'week' is invalid! It needs to be either None or in interval \[1, 48\]."):
    model.predict_species_at_location_and_time(0, 0, week=49)


def model_test_no_week(model: ModelV2M4Base):
  species = model.predict_species_at_location_and_time(
    42.5, -76.45, min_confidence=0.03)
  assert len(species) >= 252  # 255 on TFLite

  assert list(species.keys())[0] == 'Cyanocitta cristata_Blue Jay'
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9586712, decimal=3)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.06815465, decimal=3)


def model_test_using_threshold(model: ModelV2M4Base):
  species = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0.03)
  assert len(species) == 64
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9276199, decimal=3)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.035001162, decimal=3)
  assert list(species.keys())[0] == 'Cyanocitta cristata_Blue Jay'
  # is last or second last
  assert 'Larus marinus_Great Black-backed Gull' in list(species.keys())[-2:]


def model_test_using_no_threshold_returns_all_species(model: ModelV2M4Base):
  species = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0)
  assert len(species) == len(model.species) == 6522


def model_test_identical_predictions_return_same_result(model: ModelV2M4Base):
  species1 = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0)

  species2 = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0)

  assert species_prediction_is_equal(species1, species2, precision=7)


@dataclass()
class LocationTestCase():
  latitude: float = 42.5
  longitude: float = -76.45
  week: Optional[int] = None


def predict_species(test_case: LocationTestCase, model: ModelV2M4Base) -> SpeciesPrediction:
  # min_confidence=0 because otherwise the length is not always the same
  return model.predict_species_at_location_and_time(
    test_case.latitude, test_case.longitude, week=test_case.week, min_confidence=0,
  )


def create_ground_truth_test_file(model: ModelV2M4Base, path: Path):
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


def model_test_predictions_are_globally_correct(model: ModelV2M4Base, /, *, precision: int):
  with TEST_PATH.open("rb") as f:
    test_cases: List[Tuple[Dict, SpeciesPrediction]] = pickle.load(f)

  for test_case_dict, gt in tqdm(test_cases):
    test_case = LocationTestCase(**test_case_dict)
    res = predict_species(test_case, model)
    assert species_prediction_is_equal(res, gt, precision=precision)


if __name__ == "__main__":
  # global ground truth is created using protobuf CPU model
  m = ModelV2M4Protobuf(language="en_us", custom_device="/device:CPU:0")
  create_ground_truth_test_file(m, TEST_PATH)
