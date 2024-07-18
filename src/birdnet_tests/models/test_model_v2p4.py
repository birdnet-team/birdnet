import shutil
from pathlib import Path

import numpy.testing as npt

from birdnet.models.model_v2p4 import Downloader, ModelV2p4


def test_analyse_sample():
  path = Path("example/soundscape.wav")
  model = ModelV2p4()
  res = model.analyze_file(path)
  prediction_0_3 = res.predictions[(0, 3)]['Poecile atricapillus_Black-capped Chickadee']
  prediction_66_69 = res.predictions[(66, 69)]['Engine_Engine']
  npt.assert_almost_equal(prediction_0_3, 0.8140557, decimal=7)
  npt.assert_almost_equal(prediction_66_69, 0.08610276, decimal=8)


def test_get_species():
  model = ModelV2p4()
  species = model.get_species_from_location(42.5, -76.45, 4)
  assert len(species) == 64
  assert species[0] == 'Cyanocitta cristata_Blue Jay'
  assert species[63] == 'Larus marinus_Great Black-backed Gull'


def test_downloader():
  downloader = Downloader()
  shutil.rmtree(downloader.version_path)
  assert not downloader._check_model_files_exist()
  downloader.ensure_model_is_available()
  assert downloader._check_model_files_exist()
