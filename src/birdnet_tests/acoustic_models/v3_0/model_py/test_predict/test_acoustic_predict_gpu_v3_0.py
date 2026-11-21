"""On a GPU the V3.0 model returns the scores the onnx model returns on the CPU.

One case per backend and precision the README lists for a GPU. A case skips
where TensorFlow sees no GPU or its runtime has no GPU build installed.
"""

import pytest

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet_tests.acoustic_models.v3_0.model_py.conftest import (
  V3_0Backend,
  load_v3_0_for_gpu,
  onnx_fp32_prediction,
)
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  ensure_onnxruntime_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

_DEFAULT_SCORE_SHAPE = (1, 3, AcousticModelV3_0.get_n_classes())


@pytest.mark.gpu
@pytest.mark.usefixtures("tensor_float_32_off")
@pytest.mark.parametrize(
  ("backend", "precision", "max_abs_diff", "mean_abs_diff"),
  [
    # The bounds between the backends on the CPU. pb comes closest to them,
    # 2.7e-5 at the largest (RTX 3080 Ti): the TensorFloat-32 the fixture
    # cannot switch off.
    pytest.param("pb", "fp32", 1e-4, 1e-5, id="pb-fp32"),
    pytest.param("pt", "fp32", 1e-4, 1e-5, id="pt-fp32"),
    pytest.param("onnx", "fp32", 1e-4, 1e-5, id="onnx-fp32"),
    # fp16 on CUDA is 1.4e-2 and 1.1e-6 off the fp32 reference (RTX 3080 Ti),
    # on the CPU 7.0e-3 and 5.5e-7.
    pytest.param("onnx", "fp16", 5e-2, 1e-5, id="onnx-fp16"),
  ],
)
def test_v3_0_predict_on_gpu_is_close_to_onnx_on_cpu(
  backend: V3_0Backend, precision: str, max_abs_diff: float, mean_abs_diff: float
) -> None:
  ensure_onnxruntime_or_skip()
  model = load_v3_0_for_gpu(backend, precision)

  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-float("inf"),
    device="GPU",
  ) as session:
    gpu_result = session.run(TEST_FILE_SHORT)

  # The fp32 model would pass inside the fp16 bounds.
  assert gpu_result.model_precision == precision
  assert gpu_result.species_probs.shape == _DEFAULT_SCORE_SHAPE
  assert_prediction_result_is_close(
    gpu_result,
    onnx_fp32_prediction(),
    max_abs_diff=max_abs_diff,
    mean_abs_diff=mean_abs_diff,
  )
