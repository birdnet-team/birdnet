import numpy
import pytest

from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.model_loader import load
from birdnet_tests.test_files import NON_EXISTING_TEST_FILE_WAV, TEST_FILE_WAV


def get_tf_model() -> AcousticModelV2_4:
  model = load("acoustic", "2.4", "tf", lang="en_us", precision="fp32", library="tf")
  return model


def test_invalid_audio_file_path_raises_value_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Input path 'src/birdnet_tests/TEST_FILES/dummy.wav' was not found.",
  ):
    get_tf_model().predict(NON_EXISTING_TEST_FILE_WAV)


def test_litert() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=1)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=1)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=1)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)
