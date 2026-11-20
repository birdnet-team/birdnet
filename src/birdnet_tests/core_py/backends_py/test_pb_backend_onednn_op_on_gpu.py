"""``PBBackend`` names the way out where oneDNN breaks a SavedModel on a GPU.

TensorFlow's oneDNN rewrite fuses a Keras layer normalization into
``_MklLayerNorm``, which has a CPU kernel only, also where the layer is placed
on a GPU; the first call then fails with a ``NotFoundError`` (#167). The text
below is what TensorFlow 2.21.0 raised for the geo 3.0 model. That the advice
in the error holds is checked with the model, in
geo_models/v3_0/model_py/test_predict/test_geo_predict_model.py: the CPU
everywhere, the variable on a GPU.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from birdnet.acoustic.models.v3_0.pb import AcousticPBBackendFP32V3_0
from birdnet.core.backends import PBBackend
from birdnet.geo.models.v3_0.pb import GeoPBBackendFP32V3_0

_NODE = "model_38/layer_normalization_24/add"
_NO_GPU_KERNEL = (
  "Graph execution error:\n\n"
  f"Detected at node {_NODE} defined at (most recent call last):\n"
  "<stack traces unavailable>\n"
  "No registered '_MklLayerNorm' OpKernel for 'GPU' devices compatible with node "
  f"{{{{node {_NODE}}}}}\n"
  "\t.  Registered:  device='CPU'; T in [DT_HALF]\n"
  "  device='CPU'; T in [DT_BFLOAT16]\n"
  "  device='CPU'; T in [DT_FLOAT]\n\n"
  f"\t [[StatefulPartitionedCall/{_NODE}]] [Op:__inference_signature_wrapper_2082]"
)
_BATCH = np.zeros((1, 3), dtype=np.float32)


def _backend(backend_type: type[PBBackend], error: Exception) -> Any:  # noqa: ANN401
  """A backend whose signatures raise, skipping load() and the download."""

  def signature(**kwargs: object) -> dict:
    raise error

  backend = backend_type(Path("model-fp32"), "GPU", False)
  backend._logical_device_name = "CPU:0"
  backend._predict_fn = signature
  backend._encoding_fn = signature
  return backend


def _not_found(message: str) -> Exception:
  from tensorflow.errors import NotFoundError

  return NotFoundError(None, None, message)


def _invalid_argument(message: str) -> Exception:
  from tensorflow.errors import InvalidArgumentError

  return InvalidArgumentError(None, None, message)


@pytest.mark.parametrize(
  ("backend_type", "method"),
  [
    pytest.param(GeoPBBackendFP32V3_0, "predict", id="predict"),
    pytest.param(AcousticPBBackendFP32V3_0, "encode", id="encode"),
  ],
)
def test_a_onednn_op_without_gpu_kernel_names_the_way_out(
  backend_type: type[PBBackend], method: str
) -> None:
  error = _not_found(_NO_GPU_KERNEL)
  backend = _backend(backend_type, error)

  with pytest.raises(RuntimeError) as raised:
    getattr(backend, method)(_BATCH)

  message = str(raised.value)
  assert str(Path("model-fp32").absolute()) in message
  assert "into '_MklLayerNorm', an operation" in message
  assert "device='CPU'" in message
  assert "TF_ENABLE_ONEDNN_OPTS=0 before TensorFlow is imported" in message
  assert raised.value.__cause__ is error


def test_the_wording_without_quotes_is_recognized_too() -> None:
  # TensorFlow has both wordings; only the quoted one was seen raised.
  error = _not_found(_NO_GPU_KERNEL.replace("for 'GPU' devices", "for GPU devices"))
  backend = _backend(GeoPBBackendFP32V3_0, error)

  with pytest.raises(RuntimeError, match="into '_MklLayerNorm', an operation"):
    backend.predict(_BATCH)


@pytest.mark.parametrize(
  ("make_error", "message"),
  [
    pytest.param(
      _not_found,
      _NO_GPU_KERNEL.replace("_MklLayerNorm", "StringLower"),
      id="no-gpu-kernel-for-another-op",
    ),
    pytest.param(
      _not_found,
      _NO_GPU_KERNEL.replace("for 'GPU' devices", "for 'CPU' devices"),
      id="no-cpu-kernel-for-a-onednn-op",
    ),
    pytest.param(_not_found, "model-fp32/variables; No such file", id="missing-file"),
    pytest.param(_invalid_argument, _NO_GPU_KERNEL, id="another-error-class"),
    pytest.param(ValueError, _NO_GPU_KERNEL, id="not-a-tensorflow-error"),
  ],
)
def test_every_other_error_is_left_as_it_is(
  make_error: Callable[[str], Exception], message: str
) -> None:
  error = make_error(message)
  backend = _backend(GeoPBBackendFP32V3_0, error)

  with pytest.raises(type(error)) as raised:
    backend.predict(_BATCH)

  assert raised.value is error
