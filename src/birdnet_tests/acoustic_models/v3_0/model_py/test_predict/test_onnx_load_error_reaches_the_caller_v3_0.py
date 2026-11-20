"""A `ValueError` of the onnx loader in a worker reaches the caller with its text.

The worker loads the backend in its own process, where no fake reaches, so
this takes the loader's error for a runtime without any GPU provider; the one
for a GPU provider that cannot be created leaves `load_onnx_model` the same way.
"""

import pytest

from birdnet_tests.acoustic_models.v3_0.model_py.conftest import load_v3_0
from birdnet_tests.helper import ensure_no_onnxruntime_gpu_provider_or_skip
from birdnet_tests.test_files import TEST_FILE_SHORT

pytestmark = pytest.mark.no_tf


def test_the_onnx_loaders_error_in_a_worker_reaches_the_caller() -> None:
  ensure_no_onnxruntime_gpu_provider_or_skip()
  model = load_v3_0("onnx")

  with pytest.raises(RuntimeError) as raised:
    model.predict(TEST_FILE_SHORT, n_workers=1, device="GPU")

  message = str(raised.value)
  assert (
    "failed: ValueError: No ONNX Runtime GPU execution provider found" in message
  ), message
