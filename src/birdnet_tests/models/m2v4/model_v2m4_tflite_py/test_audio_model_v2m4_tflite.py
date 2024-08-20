import pytest

from birdnet.models.v2m4.model_v2m4_tflite import AVAILABLE_LANGUAGES, AudioModelV2M4TFLite


def test_invalid_language_raises_value_error():
  expectation = rf"Language 'english' is not available! Choose from: {', '.join(sorted(AVAILABLE_LANGUAGES))}."
  with pytest.raises(ValueError, match=expectation):
    AudioModelV2M4TFLite(language="english")


def test_valid_language_de_raises_no_error():
  AudioModelV2M4TFLite(language="de")
