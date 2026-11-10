"""The official 3.0 TFLite backends take their tensors from the file's signature.

Tensor indices differ between exports of the same model, so a table of them
names an unrelated tensor as soon as a release is re-exported. A file whose
signature does not describe the official model is refused instead of run.
"""

from pathlib import Path

import pytest

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendFP16V3_0,
  AcousticTFBackendFP32V3_0,
)
from birdnet.core.backends import TFBackend
from birdnet_tests.acoustic_models.custom_tflite_py.stub_interpreter import (
  StubInterpreter,
  Tensor,
)

pytestmark = pytest.mark.no_tf

_N_CLASSES = AcousticModelV3_0.get_n_classes()
_EMBEDDINGS_DIM = AcousticModelV3_0.get_embeddings_dim()
_BACKENDS = [AcousticTFBackendFP32V3_0, AcousticTFBackendFP16V3_0]


def _export(
  prediction_idx: int = 1164,
  embeddings_idx: int = 1090,
  n_classes: int = _N_CLASSES,
  embeddings_dim: int = _EMBEDDINGS_DIM,
) -> StubInterpreter:
  predictions = Tensor(prediction_idx, "StatefulPartitionedCall:1", [1, n_classes])
  embeddings = Tensor(embeddings_idx, "StatefulPartitionedCall:0", [1, embeddings_dim])
  return StubInterpreter(
    inputs=[Tensor(0, "serving_default_x:0", [1, -1])],
    outputs=[predictions, embeddings],
    signatures={
      "serving_default": {
        "inputs": ["x"],
        "outputs": {"embeddings": embeddings, "predictions": predictions},
      }
    },
  )


def _without_embeddings_output() -> StubInterpreter:
  interp = _export()
  (predictions, _) = interp.outputs
  interp.outputs = [predictions]
  interp.signatures["serving_default"]["outputs"] = {"predictions": predictions}
  return interp


def _embeddings_only_inside_the_graph() -> StubInterpreter:
  interp = _without_embeddings_output()
  interp.others = [Tensor(900, "model/GLOBAL_AVG_POOL/Mean", [1, _EMBEDDINGS_DIM])]
  return interp


def _backend(backend_type: type[TFBackend]) -> TFBackend:
  return backend_type(Path("model.file"), "CPU", False, inference_library="tflite")


@pytest.mark.parametrize("backend_type", _BACKENDS)
@pytest.mark.parametrize(
  ("prediction_idx", "embeddings_idx"), [(1164, 1090), (1546, 1472), (7, 1900)]
)
def test_indices_are_those_of_the_loaded_export(
  backend_type: type[TFBackend], prediction_idx: int, embeddings_idx: int
) -> None:
  resolved = _backend(backend_type)._resolve_tensor_indices(
    _export(prediction_idx, embeddings_idx)
  )

  assert resolved == (0, prediction_idx, embeddings_idx)


@pytest.mark.parametrize("backend_type", _BACKENDS)
@pytest.mark.parametrize(
  ("interp", "found"),
  [
    (_export(n_classes=_N_CLASSES + 1), f"found {_N_CLASSES + 1} classes"),
    (_export(embeddings_dim=1024), "an embeddings output of width 1024"),
    (_without_embeddings_output(), "no embeddings output"),
    (_embeddings_only_inside_the_graph(), "no embeddings output"),
  ],
  ids=["class count", "embeddings width", "no embeddings", "embeddings not an output"],
)
def test_a_file_that_is_not_the_official_export_is_refused(
  backend_type: type[TFBackend], interp: StubInterpreter, found: str
) -> None:
  with pytest.raises(ValueError, match="not the official acoustic 3.0") as excinfo:
    _backend(backend_type)._resolve_tensor_indices(interp)

  assert found in str(excinfo.value)
