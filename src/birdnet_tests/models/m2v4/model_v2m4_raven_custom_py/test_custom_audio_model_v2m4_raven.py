

import os

import numpy.testing as npt
import pytest

from birdnet.audio_based_prediction import predict_species_within_audio_file
from birdnet.models.v2m4.model_v2m4_raven_custom import CustomAudioModelV2M4Raven
from birdnet.models.v2m4.model_v2m4_tflite_custom import CustomAudioModelV2M4TFLite
from birdnet.types import SpeciesPredictions
from birdnet_tests.helper import TEST_FILE_WAV, TEST_FILES_DIR, convert_predictions_to_numpy

CLASSIFIER_FOLDER = TEST_FILES_DIR / "v2m4" / "custom_model_raven"
CLASSIFIER_FOLDER_TFLITE = TEST_FILES_DIR / "v2m4" / "custom_model_tflite"


def test_invalid_classifier_name_raises_value_error():
  expectation = rf"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{CLASSIFIER_FOLDER.absolute()}' doesn't contain a valid raven classifier which has the name 'abc'!"

  with pytest.raises(ValueError, match=expectation):
    CustomAudioModelV2M4Raven(CLASSIFIER_FOLDER, "abc")


def test_invalid_classifier_path_raises_value_error():
  classifier_folder = CLASSIFIER_FOLDER / "dummy"
  expectation = rf"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid raven classifier which has the name 'abc'!"

  with pytest.raises(ValueError, match=expectation):
    CustomAudioModelV2M4Raven(classifier_folder, "abc")


def test_load_custom_model():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  model = CustomAudioModelV2M4Raven(
    CLASSIFIER_FOLDER, "CustomClassifier", custom_device="/device:CPU:0")
  assert len(model.species) == 4


def test_minimum_test_soundscape_predictions_are_correct():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  model = CustomAudioModelV2M4Raven(
    CLASSIFIER_FOLDER, "CustomClassifier", custom_device="/device:CPU:0")

  res = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    custom_model=model,
  ))

  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.83264834,
    decimal=5
  )

  assert list(res[(66, 69)].keys())[0] == 'Junco hyemalis_Dark-eyed Junco'
  npt.assert_almost_equal(
    res[(66, 69)]['Junco hyemalis_Dark-eyed Junco'],
    0.19125606,
    decimal=5
  )

  assert list(res[(117, 120)].keys())[0] == 'Junco hyemalis_Dark-eyed Junco'
  npt.assert_almost_equal(
      res[(117, 120)]['Junco hyemalis_Dark-eyed Junco'],
      0.14392963,
      decimal=5
    )
  assert len(res) == 40


def test_no_sigmoid_soundscape_predictions_are_correct():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  model = CustomAudioModelV2M4Raven(
    CLASSIFIER_FOLDER, "CustomClassifier", custom_device="/device:CPU:0")

  res = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    apply_sigmoid=False,
    custom_model=model,
  ))

  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    1.604514,
    decimal=5
  )

  assert len(res) == 40


def test_no_sigmoid_soundscape_predictions_are_same_with_custom_tflite():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  model_raven = CustomAudioModelV2M4Raven(
    CLASSIFIER_FOLDER, "CustomClassifier", custom_device="/device:CPU:0")

  res_raven = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    apply_sigmoid=False,
    custom_model=model_raven,
  ))
  res_raven_np = convert_predictions_to_numpy(res_raven)

  model_tflite = CustomAudioModelV2M4TFLite(CLASSIFIER_FOLDER_TFLITE, "CustomClassifier")

  res_tflite = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    apply_sigmoid=False,
    custom_model=model_tflite,
  ))
  res_tflite_np = convert_predictions_to_numpy(res_tflite)

  npt.assert_almost_equal(res_raven_np[0][2][0], -2.4397974, decimal=5)
  npt.assert_array_almost_equal(
    res_raven_np[0],
    res_tflite_np[0],
    decimal=3
  )
  assert res_raven_np[1] == res_tflite_np[1]
  assert res_raven_np[2] == res_tflite_np[2]


def test_sigmoid_soundscape_predictions_are_same_with_custom_tflite():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  model_raven = CustomAudioModelV2M4Raven(
    CLASSIFIER_FOLDER, "CustomClassifier", custom_device="/device:CPU:0")

  res_raven = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    apply_sigmoid=True,
    custom_model=model_raven,
  ))
  res_raven_np = convert_predictions_to_numpy(res_raven)

  model_tflite = CustomAudioModelV2M4TFLite(CLASSIFIER_FOLDER_TFLITE, "CustomClassifier")

  res_tflite = SpeciesPredictions(predict_species_within_audio_file(
    TEST_FILE_WAV,
    min_confidence=0,
    apply_sigmoid=True,
    custom_model=model_tflite,
  ))
  res_tflite_np = convert_predictions_to_numpy(res_tflite)

  npt.assert_almost_equal(res_raven_np[0][0][0], 0.03020441, decimal=5)
  npt.assert_array_almost_equal(
    res_raven_np[0],
    res_tflite_np[0],
    decimal=3
  )
  assert res_raven_np[1] == res_tflite_np[1]
  assert res_raven_np[2] == res_tflite_np[2]
