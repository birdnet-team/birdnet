from pathlib import Path
from typing import Set

import numpy.testing as npt
import pytest

from birdnet.models.model_v2m4 import ModelV2M4
from birdnet.types import Species

TEST_FILES_DIR = Path("src/birdnet_tests/test_files")
TEST_FILE_WAV = TEST_FILES_DIR / "soundscape.wav"


@pytest.fixture()
def model():
  model = ModelV2M4(language="en_us")
  return model


def test_soundscape_predictions_are_correct(model: ModelV2M4):
  res = model.predict_species_within_audio_file(TEST_FILE_WAV, min_confidence=0)

  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140557, decimal=7)

  npt.assert_almost_equal(res[(66, 69)]['Engine_Engine'],
                          0.08610276, decimal=8)
  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  assert list(res[(66, 69)].keys())[0] == 'Engine_Engine'


def test_identical_predictions_return_same_result(model: ModelV2M4):
  res1 = model.predict_species_within_audio_file(TEST_FILE_WAV, min_confidence=0)

  res2 = model.predict_species_within_audio_file(TEST_FILE_WAV, min_confidence=0)

  npt.assert_almost_equal(
    res1[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140557, decimal=7)

  npt.assert_almost_equal(res1[(66, 69)]['Engine_Engine'],
                          0.08610276, decimal=8)
  assert list(res1[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  assert list(res1[(66, 69)].keys())[0] == 'Engine_Engine'

  npt.assert_almost_equal(
    res2[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140557, decimal=7)

  npt.assert_almost_equal(res2[(66, 69)]['Engine_Engine'],
                          0.08610276, decimal=8)
  assert list(res2[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  assert list(res2[(66, 69)].keys())[0] == 'Engine_Engine'


def test_invalid_audio_file_path_raises_value_error(model: ModelV2M4):
  # Test für ungültigen Audio-Dateipfad
  with pytest.raises(ValueError, match=r"Value for 'audio_file' is invalid! It needs to be a path to an existing audio file."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILES_DIR / "dummy.wav"
    )


def test_invalid_batch_size_raises_value_error(model: ModelV2M4):
  # Test für ungültige Batch-Größe
  with pytest.raises(ValueError, match=r"Value for 'batch_size' is invalid! It needs to be larger than zero."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        batch_size=0
    )


def test_invalid_file_splitting_duration_raises_value_error(model: ModelV2M4):
  # Test für ungültige Datei-Aufteilungsdauer
  with pytest.raises(ValueError, match=r"Value for 'file_splitting_duration_s' is invalid! It needs to be larger than or equal to 3.0."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        file_splitting_duration_s=2
    )


def test_invalid_min_confidence_raises_value_error(model: ModelV2M4):
  # Test für ungültigen Min-Konfidenzwert
  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0, 1.0\)."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        min_confidence=-0.1
    )

  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0, 1.0\)."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        min_confidence=1.1
    )


def test_invalid_bandpass_fmin_raises_value_error(model: ModelV2M4):
  # Test für ungültigen Bandpass-Filter fmin-Wert
  with pytest.raises(ValueError, match=r"Value for 'bandpass_fmin' is invalid! It needs to be larger than zero."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        bandpass_fmin=-1
    )


def test_invalid_bandpass_fmax_raises_value_error(model: ModelV2M4):
  # Test für ungültigen Bandpass-Filter fmax-Wert
  with pytest.raises(ValueError, match=r"Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        bandpass_fmin=500,
        bandpass_fmax=400
    )


def test_invalid_sigmoid_sensitivity_raises_value_error(model: ModelV2M4):
  # Test für ungültigen Sigmoid-Sensitivitätswert
  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!"):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=None
    )

  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval \[0.5, 1.5\]."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=0.4
    )

  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval \[0.5, 1.5\]."):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=1.6
    )


def test_invalid_species_filter_raises_value_error(model: ModelV2M4):
  # Test für ungültigen Art-Filter
  invalid_filter_species: Set[Species] = {"species"}
  with pytest.raises(ValueError, match=rf"At least one species defined in 'filter_species' is invalid! They need to be known species, e.g., {', '.join(model._species_list[:3])}"):
    model.predict_species_within_audio_file(
        audio_file=TEST_FILE_WAV,
        filter_species=invalid_filter_species
    )