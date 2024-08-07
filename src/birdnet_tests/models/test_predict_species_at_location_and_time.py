import numpy.testing as npt
import pytest

from birdnet.models.model_v2m4_base import ModelV2M4Base


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
  assert len(species) == 6522
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9276199, decimal=3)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.035001162, decimal=3)


def model_test_identical_predictions_return_same_result(model: ModelV2M4Base):
  species1 = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0)

  species2 = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0)

  assert len(species1) == len(species2) == 6522
  npt.assert_almost_equal(species1['Cyanocitta cristata_Blue Jay'],
                          species2['Cyanocitta cristata_Blue Jay'], decimal=7)  # 0.9276199
  npt.assert_almost_equal(species1['Larus marinus_Great Black-backed Gull'],
                          species2['Larus marinus_Great Black-backed Gull'], decimal=7)  # 0.035001162
