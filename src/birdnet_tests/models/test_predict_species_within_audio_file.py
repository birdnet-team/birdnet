from pathlib import Path

import numpy.testing as npt
import pytest

from birdnet.models.model_v2m4_base import ModelV2M4Base

TEST_FILES_DIR = Path("src/birdnet_tests/test_files")
# Duration: 120s
TEST_FILE_WAV = TEST_FILES_DIR / "soundscape.wav"


class DummyModel(ModelV2M4Base):
  pass


@pytest.fixture(name="model")
def get_model():
  model = DummyModel(language="en_us")
  return model


def test_invalid_audio_file_path_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'audio_file' is invalid! It needs to be a path to an existing audio file."):
    model.predict_species_within_audio_file(
        TEST_FILES_DIR / "dummy.wav"
    )


def test_invalid_batch_size_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'batch_size' is invalid! It needs to be larger than zero."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        batch_size=0
    )


def test_invalid_min_confidence_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        min_confidence=-0.1
    )

  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        min_confidence=1.1
    )


def test_invalid_chunk_overlap_s_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'chunk_overlap_s' is invalid! It needs to be in interval \[0.0, 3.0\)"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        chunk_overlap_s=-1,
    )

  with pytest.raises(ValueError, match=r"Value for 'chunk_overlap_s' is invalid! It needs to be in interval \[0.0, 3.0\)"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        chunk_overlap_s=4.0,
    )


def test_invalid_bandpass_fmin_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'bandpass_fmin' is invalid! It needs to be larger than zero."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        bandpass_fmin=-1
    )


def test_invalid_bandpass_fmax_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        bandpass_fmin=500,
        bandpass_fmax=400
    )


def test_invalid_sigmoid_sensitivity_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=None
    )

  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval \[0.5, 1.5\]."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=0.4
    )

  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval \[0.5, 1.5\]."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=1.6
    )


def model_test_soundscape_predictions_are_correct(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140561, decimal=5)

  assert list(res[(66, 69)].keys())[0] == 'Engine_Engine'
  npt.assert_almost_equal(res[(66, 69)]['Engine_Engine'],
                          0.0861028, decimal=5)
  assert len(res) == 120 / 3 == 40


def model_test_soundscape_predictions_batch_size_4_are_correct(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, batch_size=4)

  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8140561, decimal=5)

  assert list(res[(66, 69)].keys())[0] == 'Engine_Engine'
  npt.assert_almost_equal(res[(66, 69)]['Engine_Engine'],
                          0.0861028, decimal=5)
  assert len(res) == 120 / 3 == 40


def model_test_identical_predictions_return_same_result(model: ModelV2M4Base):
  res1 = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  res2 = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  npt.assert_almost_equal(
    res1[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
      res2[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'], decimal=5)  # 0.8140561

  npt.assert_almost_equal(res1[(66, 69)]['Engine_Engine'],
                          res2[(66, 69)]['Engine_Engine'], decimal=5)  # 0.0861028
  assert list(res1[(0, 3)].keys())[0] == list(res2[(0, 3)].keys())[
      0]  # 'Poecile atricapillus_Black-capped Chickadee'
  assert list(res1[(66, 69)].keys())[0] == list(res2[(66, 69)].keys())[0]  # 'Engine_Engine'
