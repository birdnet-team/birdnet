import pytest

from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet_tests.test_files import NON_EXISTING_TEST_FILE_WAV


@pytest.fixture(name="model")
def provide_model_to_tests():
  model = AcousticTFModelV2_4.load(lang="en_us", precision="fp32", library="tf")
  return model


def test_invalid_audio_file_path_raises_value_error(model: AcousticTFModelV2_4):
  with pytest.raises(
    ValueError,
    match=r"Input path 'src/birdnet_tests/TEST_FILES/dummy.wav' was not found.",
  ):
    model.predict(NON_EXISTING_TEST_FILE_WAV)
