import pytest

from birdnet.models.v2m4.model_v2m4_protobuf import AVAILABLE_LANGUAGES, MetaModelV2M4Protobuf


def test_invalid_language_raises_value_error():
  expectation = rf"Language 'english' is not available! Choose from: {', '.join(sorted(AVAILABLE_LANGUAGES))}."
  with pytest.raises(ValueError, match=expectation):
    MetaModelV2M4Protobuf(language="english")


def test_valid_language_de_raises_no_error():
  MetaModelV2M4Protobuf(language="de")
