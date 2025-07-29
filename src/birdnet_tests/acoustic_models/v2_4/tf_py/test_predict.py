import pytest

from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet_tests.test_files import NON_EXISTING_TEST_FILE_WAV, TEST_FILES_DIR


def test():
  model = AcousticTFModelV2_4.load(lang="en_us", precision="fp32", library="tf")
  model.predict()
  pass


def test_invalid_audio_file_path_raises_value_error(model: AcousticTFModelV2_4):
  with pytest.raises(
    ValueError,
    match=r"Value for 'audio_file' is invalid! It needs to be a path to an existing audio file.",
  ):
    model.predict(NON_EXISTING_TEST_FILE_WAV)
