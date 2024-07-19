import tempfile
from pathlib import Path

import numpy.testing as npt
import pytest

from birdnet.models.model_v2p4 import Downloader, ModelV2p4


def test_soundscape_predictions_are_correct():
  test_files_dir = Path("src/birdnet_tests/test_files")
  path = test_files_dir / "soundscape.wav"
  model = ModelV2p4()

  res = model.predict_species_in_audio_file(path, min_confidence=0)

  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140557, decimal=7)

  npt.assert_almost_equal(res[(66, 69)]['Engine_Engine'],
                          0.08610276, decimal=8)
  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  assert list(res[(66, 69)].keys())[0] == 'Engine_Engine'
