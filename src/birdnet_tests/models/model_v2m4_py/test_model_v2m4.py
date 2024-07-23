import pytest

from birdnet.models.model_v2m4 import AVAILABLE_LANGUAGES, ModelV2M4


def test_invalid_language_raises_value_error():
  expectation = rf"Language 'english' is not available! Choose from: {', '.join(sorted(AVAILABLE_LANGUAGES))}."
  with pytest.raises(ValueError, match=expectation):
    ModelV2M4(language="english")
