from pathlib import Path

import numpy.testing as npt
import pytest

from birdnet.models.model_v2m4_tflite_custom import CustomModelV2M4TFLite
from birdnet_tests.helper import TEST_FILES_DIR

# Duration: 120s
TEST_FILE_WAV = TEST_FILES_DIR / "soundscape.wav"


def test_invalid_classifier_name_raises_value_error():
  classifier_folder = Path("src/birdnet_tests/test_files/custom_model_v2m4_tflite")
  expectation = rf"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid TFLite classifier which has the name 'abc'!"

  with pytest.raises(ValueError, match=expectation):
    CustomModelV2M4TFLite(classifier_folder, "abc")


def test_invalid_classifier_path_raises_value_error():
  classifier_folder = Path("src/birdnet_tests/test_files/custom_model_v2m4_tflite_dummy")
  expectation = rf"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid TFLite classifier which has the name 'abc'!"

  with pytest.raises(ValueError, match=expectation):
    CustomModelV2M4TFLite(classifier_folder, "abc")


def test_load_custom_model():
  classifier_folder = Path("src/birdnet_tests/test_files/custom_model_v2m4_tflite")
  model = CustomModelV2M4TFLite(classifier_folder, "CustomClassifier")
  assert len(model.species) == 4


def test_minimum_test_soundscape_predictions_are_correct():
  classifier_folder = Path("src/birdnet_tests/test_files/custom_model_v2m4_tflite")
  model = CustomModelV2M4TFLite(classifier_folder, "CustomClassifier")

  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.7648012,
    decimal=6
  )

  assert list(res[(66, 69)].keys())[0] == 'Junco hyemalis_Dark-eyed Junco'
  npt.assert_almost_equal(
    res[(66, 69)]['Junco hyemalis_Dark-eyed Junco'],
    0.20300059,
    decimal=6
  )

  assert list(res[(117, 120)].keys())[0] == 'Cyanocitta cristata_Blue Jay'
  npt.assert_almost_equal(
      res[(117, 120)]['Cyanocitta cristata_Blue Jay'],
      0.14364028,
      decimal=6
    )
  assert len(res) == 40
