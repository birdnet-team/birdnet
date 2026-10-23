import pytest

from birdnet.acoustic.models.custom_tflite import (
  CustomTFLiteInfo,
  CustomTFLiteLayout,
  describe_custom_tflite_layout,
)

pytestmark = pytest.mark.no_tf


def _layout(
  n_species: int,
  input_samples: int | None,
  embeddings_dim: int | None,
) -> CustomTFLiteLayout:
  return CustomTFLiteLayout(
    input_idx=0,
    prediction_idx=5,
    embeddings_idx=None if embeddings_dim is None else 4,
    embeddings_is_output=False,
    n_species=n_species,
    input_samples=input_samples,
    embeddings_dim=embeddings_dim,
  )


def test_2_4_replace_from_the_input_length_and_embeddings() -> None:
  assert describe_custom_tflite_layout(_layout(3, 144_000, 1024)) == CustomTFLiteInfo(
    "2.4", "replace", 3, False, True, 144_000
  )


def test_3_0_append_from_the_species_count() -> None:
  assert describe_custom_tflite_layout(_layout(11_563, 96_000, 1280)) == (
    CustomTFLiteInfo("3.0", "append", 11_563, True, True, 96_000)
  )


def test_a_dynamic_input_length_is_reported_as_none() -> None:
  assert describe_custom_tflite_layout(_layout(3, None, 1280)).input_samples is None


def test_exactly_the_official_count_is_still_replace() -> None:
  assert describe_custom_tflite_layout(_layout(6522, 144_000, 1024)).mode == "replace"


def test_version_from_the_embeddings_when_the_input_is_dynamic() -> None:
  assert describe_custom_tflite_layout(_layout(3, None, 1280)).version == "3.0"


def test_version_from_the_input_when_there_are_no_embeddings() -> None:
  info = describe_custom_tflite_layout(_layout(3, 96_000, None))
  assert info.version == "3.0"
  assert not info.supports_encoding


def test_nothing_to_derive_the_version_from_is_rejected() -> None:
  with pytest.raises(ValueError, match=r"Neither the input length"):
    describe_custom_tflite_layout(_layout(3, None, None))


def test_a_detached_head_is_named_as_such() -> None:
  with pytest.raises(ValueError, match=r"takes 1280 samples.*detached classifier head"):
    describe_custom_tflite_layout(_layout(3, 1280, None))


def test_an_unknown_input_length_is_rejected() -> None:
  with pytest.raises(
    ValueError, match=r"takes 48000 samples, but a BirdNET classifier"
  ):
    describe_custom_tflite_layout(_layout(3, 48_000, None))


def test_an_unknown_embeddings_width_is_rejected() -> None:
  with pytest.raises(ValueError, match=r"embeddings are 512 wide"):
    describe_custom_tflite_layout(_layout(3, 144_000, 512))


def test_disagreeing_input_and_embeddings_are_rejected() -> None:
  with pytest.raises(ValueError, match=r"144000 samples \(BirdNET 2\.4\).*1280 wide"):
    describe_custom_tflite_layout(_layout(3, 144_000, 1280))
