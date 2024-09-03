import numpy.testing as npt
import pytest

from birdnet.audio_based_prediction import predict_species_within_audio_file
from birdnet.models.v2m4.model_v2m4_tflite_custom import CustomAudioModelV2M4TFLite
from birdnet.types import SpeciesPredictions
from birdnet_tests.helper import TEST_FILE_WAV, TEST_FILES_DIR

CLASSIFIER_FOLDER = TEST_FILES_DIR / "v2m4" / "custom_model_tflite"


def test_invalid_classifier_name_raises_value_error():
  expectation = rf"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{CLASSIFIER_FOLDER.absolute()}' doesn't contain a valid TFLite classifier which has the name 'abc'!"

  with pytest.raises(ValueError, match=expectation):
    CustomAudioModelV2M4TFLite(CLASSIFIER_FOLDER, "abc")


def test_invalid_classifier_path_raises_value_error():
  classifier_folder = CLASSIFIER_FOLDER / "dummy"
  expectation = rf"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid TFLite classifier which has the name 'abc'!"

  with pytest.raises(ValueError, match=expectation):
    CustomAudioModelV2M4TFLite(classifier_folder, "abc")


def test_load_custom_model():
  model = CustomAudioModelV2M4TFLite(CLASSIFIER_FOLDER, "CustomClassifier")
  assert len(model.species) == 4


def test_minimum_test_soundscape_predictions_are_correct():
  model = CustomAudioModelV2M4TFLite(CLASSIFIER_FOLDER, "CustomClassifier")

  res = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    custom_model=model,
  ))

  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.8326453,
    decimal=5
  )

  assert list(res[(66, 69)].keys())[0] == 'Junco hyemalis_Dark-eyed Junco'
  npt.assert_almost_equal(
    res[(66, 69)]['Junco hyemalis_Dark-eyed Junco'],
    0.19126873,
    decimal=5
  )

  assert list(res[(117, 120)].keys())[0] == 'Junco hyemalis_Dark-eyed Junco'
  npt.assert_almost_equal(
      res[(117, 120)]['Junco hyemalis_Dark-eyed Junco'],
      0.14393075,
      decimal=5
    )
  assert len(res) == 40


def test_no_sigmoid_soundscape_predictions_are_correct():
  model = CustomAudioModelV2M4TFLite(CLASSIFIER_FOLDER, "CustomClassifier")

  res = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    apply_sigmoid=False,
    custom_model=model,
  ))

  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    1.6044917,
    decimal=5
  )

  assert len(res) == 40
