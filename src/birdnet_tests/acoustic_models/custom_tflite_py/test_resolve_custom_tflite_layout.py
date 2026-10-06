import pytest

from birdnet.acoustic.models.custom_tflite import (
  LEGACY_V2_4_PREDICTION_INDICES,
  resolve_custom_tflite_layout,
)
from birdnet_tests.acoustic_models.custom_tflite_py.stub_interpreter import (
  StubInterpreter,
  Tensor,
  audio_input,
  v2_4_export,
  v3_0_export,
)

pytestmark = pytest.mark.no_tf


def test_signature_names_pick_the_tensors_of_a_3_0_export() -> None:
  layout = resolve_custom_tflite_layout(v3_0_export(3))
  assert layout.input_idx == 0
  assert layout.prediction_idx == 1004
  assert layout.embeddings_idx == 1002
  assert layout.embeddings_is_output
  assert layout.n_species == 3
  assert layout.input_samples == 96_000
  assert layout.embeddings_dim == 1280


def test_species_count_comes_from_the_prediction_output_wherever_it_is_listed() -> None:
  interp = v3_0_export(11_563)
  interp.outputs.reverse()  # embeddings first, as the signature lists them
  layout = resolve_custom_tflite_layout(interp)
  assert layout.n_species == 11_563
  assert layout.embeddings_dim == 1280


def test_single_signature_output_and_pooling_tensor_of_a_2_4_export() -> None:
  layout = resolve_custom_tflite_layout(v2_4_export(6525, prediction_idx=549))
  assert layout.prediction_idx == 549
  assert layout.embeddings_idx == 548
  assert not layout.embeddings_is_output
  assert layout.n_species == 6525
  assert layout.input_samples == 144_000
  assert layout.embeddings_dim == 1024


def test_single_output_without_signature() -> None:
  interp = v2_4_export(4)
  interp.signatures = {}
  layout = resolve_custom_tflite_layout(interp)
  assert layout.prediction_idx == 545
  assert layout.embeddings_idx == 544


def test_legacy_index_breaks_the_tie_between_unnamed_outputs() -> None:
  interp = StubInterpreter(
    inputs=[audio_input(144_000)],
    outputs=[Tensor(547, "x", [-1, 1024]), Tensor(550, "y", [-1, 6525])],
  )
  layout = resolve_custom_tflite_layout(interp, LEGACY_V2_4_PREDICTION_INDICES.keys())
  assert layout.prediction_idx == 550
  assert layout.n_species == 6525


def test_unnamed_outputs_without_a_legacy_index_are_rejected() -> None:
  interp = StubInterpreter(
    inputs=[audio_input(144_000)],
    outputs=[Tensor(10, "x", [-1, 1024]), Tensor(11, "y", [-1, 3])],
  )
  with pytest.raises(ValueError, match=r"Cannot tell which of the 2 output tensors"):
    resolve_custom_tflite_layout(interp, LEGACY_V2_4_PREDICTION_INDICES.keys())


def test_no_embeddings_tensor_leaves_the_embeddings_unset() -> None:
  interp = v2_4_export(3)
  interp.others = []
  layout = resolve_custom_tflite_layout(interp)
  assert layout.embeddings_idx is None
  assert layout.embeddings_dim is None


def test_a_dynamic_input_length_is_reported_as_unknown() -> None:
  interp = v3_0_export(3)
  interp.inputs = [Tensor(0, "x", [1, -1])]
  assert resolve_custom_tflite_layout(interp).input_samples is None


def test_two_inputs_are_rejected() -> None:
  interp = v3_0_export(3)
  interp.inputs = [audio_input(96_000), Tensor(1, "extra", [-1, 10])]
  with pytest.raises(ValueError, match=r"Expected one input tensor, found 2"):
    resolve_custom_tflite_layout(interp)
