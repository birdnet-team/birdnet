import re

import pytest

from birdnet.model_loader import load
from birdnet_tests.test_files import NON_EXISTING_TEST_FILE_WAV


def test_invalid_audio_file_path_raises_value_error() -> None:
  model = load(
    "acoustic", "2.4", "tf", lang="en_us", precision="fp32", library="tflite"
  )
  with pytest.raises(
    ValueError,
    match=re.escape(f"Input path '{NON_EXISTING_TEST_FILE_WAV}' was not found."),
  ):
    model.predict(NON_EXISTING_TEST_FILE_WAV)
