"""On a GPU the V3.0 model returns the embeddings the onnx model returns on the CPU.

One case per backend and precision the README lists for a GPU. A case skips
where TensorFlow sees no GPU or its runtime has no GPU build installed.
"""

import pytest

from birdnet_tests.acoustic_models.v3_0.model_py.conftest import (
  V3_0Backend,
  load_v3_0_for_gpu,
  onnx_fp32_encoding,
)
from birdnet_tests.helper import (
  assert_encoding_result_is_close,
  ensure_onnxruntime_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

_DEFAULT_EMB_SHAPE = (1, 3, 1280)


@pytest.mark.gpu
@pytest.mark.usefixtures("tensor_float_32_off")
@pytest.mark.parametrize(
  ("backend", "precision", "max_abs_diff", "mean_abs_diff"),
  [
    pytest.param("pb", "fp32", 1e-4, 1e-5, id="pb-fp32"),
    pytest.param("pt", "fp32", 1e-4, 1e-5, id="pt-fp32"),
    pytest.param("onnx", "fp32", 1e-4, 1e-5, id="onnx-fp32"),
    # fp16 on CUDA is 1.6e-2 and 2.0e-4 off the fp32 reference (RTX 3080 Ti),
    # on the CPU 6.7e-3 and 9.7e-5. Tighter than the CPU tests' 2e-1 and 2e-3,
    # under which a reference shifted by 1e-3 passes.
    pytest.param("onnx", "fp16", 5e-2, 6e-4, id="onnx-fp16"),
  ],
)
def test_v3_0_encode_on_gpu_is_close_to_onnx_on_cpu(
  backend: V3_0Backend, precision: str, max_abs_diff: float, mean_abs_diff: float
) -> None:
  ensure_onnxruntime_or_skip()
  model = load_v3_0_for_gpu(backend, precision)

  with model.encode_session(n_workers=1, device="GPU") as session:
    gpu_result = session.run(TEST_FILE_SHORT)

  # The fp32 model would pass inside the fp16 bounds.
  assert gpu_result.model_precision == precision
  assert gpu_result.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert_encoding_result_is_close(
    gpu_result,
    onnx_fp32_encoding(),
    max_abs_diff=max_abs_diff,
    mean_abs_diff=mean_abs_diff,
  )
