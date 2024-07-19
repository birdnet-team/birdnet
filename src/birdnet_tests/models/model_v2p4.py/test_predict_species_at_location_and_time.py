import tempfile
from pathlib import Path

import numpy.testing as npt
import pytest

from birdnet.models.model_v2p4 import Downloader, ModelV2p4


def test_raises_error_on_week_zero():
  model = ModelV2p4()
  with pytest.raises(ValueError):
    model.predict_species_at_location_and_time(
        42.5, -76.45, week=0)


def test_no_week():
  model = ModelV2p4()
  species = model.predict_species_at_location_and_time(
    42.5, -76.45, week=-1, min_confidence=0.03)
  assert len(species) == 255

  assert list(species.keys())[0] == 'Cyanocitta cristata_Blue Jay'
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9586712, decimal=7)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.06815465, decimal=7)


def test_using_threshold():
  model = ModelV2p4()
  species = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0.03)
  assert len(species) == 64
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9276199, decimal=7)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.035001162, decimal=8)
  assert list(species.keys())[0] == 'Cyanocitta cristata_Blue Jay'
  assert list(species.keys())[-1] == 'Larus marinus_Great Black-backed Gull'


def test_using_no_threshold():
  model = ModelV2p4()
  species = model.predict_species_at_location_and_time(
    42.5, -76.45, week=4, min_confidence=0)
  assert len(species) == 6522
  npt.assert_almost_equal(species['Cyanocitta cristata_Blue Jay'], 0.9276199, decimal=7)
  npt.assert_almost_equal(species['Larus marinus_Great Black-backed Gull'], 0.035001162, decimal=8)
