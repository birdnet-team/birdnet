"""The official 3.0 TFLite backends take their tensors from the file's signature.

Tensor indices differ between exports of the same model, so a table of them
names an unrelated tensor as soon as a release is re-exported.
"""

from pathlib import Path

import pytest

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


def _export(prediction_idx: int, embeddings_idx: int) -> StubInterpreter:
  predictions = Tensor(prediction_idx, "StatefulPartitionedCall:1", [1, 5])
  embeddings = Tensor(embeddings_idx, "StatefulPartitionedCall:0", [1, 1280])
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


@pytest.mark.parametrize(
  "backend_type", [AcousticTFBackendFP32V3_0, AcousticTFBackendFP16V3_0]
)
@pytest.mark.parametrize(
  ("prediction_idx", "embeddings_idx"), [(1164, 1090), (1546, 1472), (7, 1900)]
)
def test_indices_are_those_of_the_loaded_export(
  backend_type: type[TFBackend], prediction_idx: int, embeddings_idx: int
) -> None:
  backend = backend_type(Path("model.file"), "CPU", False, inference_library="tflite")

  resolved = backend._resolve_tensor_indices(_export(prediction_idx, embeddings_idx))

  assert resolved == (0, prediction_idx, embeddings_idx)
