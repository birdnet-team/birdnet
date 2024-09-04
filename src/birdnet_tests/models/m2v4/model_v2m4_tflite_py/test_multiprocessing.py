from pathlib import Path

import numpy.testing as npt
import pytest

from birdnet.audio_based_prediction_mp import predict_species_within_audio_files_mp
from birdnet.models.v2m4.model_v2m4_tflite import AudioModelV2M4TFLite
from birdnet_tests.models.m2v4.test_predict_species_within_audio_file import TEST_FILE_WAV


@pytest.fixture(name="model")
def provide_model_to_tests():
  return get_model()


def get_model():
  model = AudioModelV2M4TFLite(language="en_us")
  return model


def test_invalid_path_is_ignored(model: AudioModelV2M4TFLite):
  res = list(predict_species_within_audio_files_mp(
    [Path(str(TEST_FILE_WAV) + "dummy"), Path(str(TEST_FILE_WAV) + "dummy")],
    min_confidence=0,
    custom_model=model,
    custom_n_jobs=2,
  ))
  assert len(res) == 0


def test_minimum_test_soundscape_predictions_are_correct(model: AudioModelV2M4TFLite):
  res = list(predict_species_within_audio_files_mp(
    [TEST_FILE_WAV, TEST_FILE_WAV],
    min_confidence=0,
    custom_model=model,
    custom_n_jobs=2,
  ))

  assert len(res) == 2

  first_res = res[0]
  first_path = first_res[0]
  first_pred = first_res[1]
  assert first_path == TEST_FILE_WAV

  assert list(first_pred[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    first_pred[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140561,
    decimal=5
  )

  assert list(first_pred[(66, 69)].keys())[0] == 'Engine_Engine'
  npt.assert_almost_equal(
    first_pred[(66, 69)]['Engine_Engine'],
    0.0861028,
    decimal=5
  )

  assert list(first_pred[(117, 120)].keys())[0] == 'Spinus tristis_American Goldfinch'
  npt.assert_almost_equal(
      first_pred[(117, 120)]['Spinus tristis_American Goldfinch'],
      0.39239582,
      decimal=5
    )

  assert len(first_pred) == 40
  second_pred = res[1][1]
  assert first_pred == second_pred
