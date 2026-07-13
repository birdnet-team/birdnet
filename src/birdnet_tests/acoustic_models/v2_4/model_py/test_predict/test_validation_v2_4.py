import re

import pytest

from birdnet.model_loader import load
from birdnet_tests.test_files import NON_EXISTING_TEST_FILE_WAV, TEST_FILE_SHORT


def test_invalid_audio_file_path_raises_value_error() -> None:
  model = load(
    "acoustic", "2.4", "tf", lang="en_us", precision="fp32", library="tflite"
  )
  with pytest.raises(
    ValueError,
    match=re.escape(f"Input path '{NON_EXISTING_TEST_FILE_WAV}' was not found."),
  ):
    model.predict(NON_EXISTING_TEST_FILE_WAV)


def test_sigmoid_and_softmax_raises_value_error() -> None:
  model = load(
    "acoustic", "2.4", "tf", lang="en_us", precision="fp32", library="tflite"
  )
  with pytest.raises(
    ValueError,
    match=re.escape("apply_sigmoid and apply_softmax cannot both be True"),
  ):
    model.predict(TEST_FILE_SHORT, apply_sigmoid=True, apply_softmax=True)


def test_softmax_with_default_sigmoid_raises_value_error() -> None:
  # apply_sigmoid defaults to True for v2.4, so it has to be disabled explicitly
  model = load(
    "acoustic", "2.4", "tf", lang="en_us", precision="fp32", library="tflite"
  )
  with pytest.raises(
    ValueError,
    match=re.escape("apply_sigmoid and apply_softmax cannot both be True"),
  ):
    model.predict(TEST_FILE_SHORT, apply_softmax=True)


def test_session_sigmoid_and_softmax_raises_value_error() -> None:
  model = load(
    "acoustic", "2.4", "tf", lang="en_us", precision="fp32", library="tflite"
  )
  with pytest.raises(
    ValueError,
    match=re.escape("apply_sigmoid and apply_softmax cannot both be True"),
  ):
    model.predict_session(apply_sigmoid=True, apply_softmax=True)
