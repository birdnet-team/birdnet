import numpy
import pytest

from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILE_LONG


def predict_test_case(
  model: AcousticModelV2_4,
  n_workers: int,
  device: str,
) -> float:
  res = model.predict(
    TEST_FILE_LONG,
    n_workers=n_workers,
    device=device,
    apply_sigmoid=True,
    top_k=None,
    default_confidence_threshold=0,
  )
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 6522)
  return mean


@pytest.mark.repro
def test_pb_cpu_fp32() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.00013913162)


@pytest.mark.repro
@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  mean = predict_test_case(model, n_workers=1, device="GPU")
  # last decimal differs on different runs
  numpy.testing.assert_almost_equal(mean, 0.0623320, decimal=7)


@pytest.mark.repro
def test_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.00013913159)


@pytest.mark.repro
def test_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.00013931823)


@pytest.mark.repro
def test_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.0001324143)


@pytest.mark.repro
@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.00013913159)


@pytest.mark.repro
@pytest.mark.litert
def test_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.00013931823)


@pytest.mark.repro
@pytest.mark.litert
def test_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  mean = predict_test_case(model, n_workers=4, device="CPU")
  numpy.testing.assert_equal(mean, 0.00013151321)
