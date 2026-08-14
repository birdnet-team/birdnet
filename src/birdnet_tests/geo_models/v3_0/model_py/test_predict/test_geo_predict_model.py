import numpy as np
import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_onnxruntime_or_skip,
  ensure_tf_2_19_or_2_18,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
  geo_v3_0_litert_not_supported_skip,
)

# --- FP32 ---


@pytest.mark.litert
def test_litert_fp32() -> None:
  geo_v3_0_litert_not_supported_skip()

  model = load("geo", "3.0", "tf", precision="fp32", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_tf_fp32() -> None:
  ensure_tf_2_19_or_2_18()

  model = load("geo", "3.0", "tf", precision="fp32", library="tflite")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


@pytest.mark.litert
def test_litert_fp32_half() -> None:
  geo_v3_0_litert_not_supported_skip()

  model = load("geo", "3.0", "tf", precision="fp32", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=True)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_tf_fp32_half() -> None:
  ensure_tf_2_19_or_2_18()

  model = load("geo", "3.0", "tf", precision="fp32", library="tflite")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=True)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


# --- FP16 ---


@pytest.mark.litert
def test_litert_fp16() -> None:
  geo_v3_0_litert_not_supported_skip()

  model = load("geo", "3.0", "tf", precision="fp16", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp16"


def test_tf_fp16() -> None:
  ensure_tf_2_19_or_2_18()

  model = load("geo", "3.0", "tf", precision="fp16", library="tflite")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp16"


# --- INT8 ---


@pytest.mark.litert
def test_litert_int8() -> None:
  geo_v3_0_litert_not_supported_skip()

  model = load("geo", "3.0", "tf", precision="int8", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "int8"


def test_tf_int8() -> None:
  ensure_tf_2_19_or_2_18()

  model = load("geo", "3.0", "tf", precision="int8", library="tflite")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "int8"


# --- GPU ---


@pytest.mark.gpu
def test_tf_fp32_gpu() -> None:
  ensure_gpu_or_skip()
  ensure_tf_2_19_or_2_18()

  model = load("geo", "3.0", "tf", precision="fp32", library="tflite")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=False, device="GPU"
  )

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


# --- PB ---


def test_pb_cpu() -> None:
  model = load("geo", "3.0", "pb", precision="fp32")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


@pytest.mark.gpu
def test_pb_gpu() -> None:
  ensure_gpu_or_skip()

  model = load("geo", "3.0", "pb", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=False, device="GPU"
  )

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_pb_cpu_half() -> None:
  model = load("geo", "3.0", "pb", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=True, device="CPU"
  )

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


# --- PT ---


def test_pt_fp32() -> None:
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()

  model = load("geo", "3.0", "pt", precision="fp32")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_pt_year_round() -> None:
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()

  model = load("geo", "3.0", "pt", precision="fp32")
  result = model.predict(20, 50, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == -1
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_pt_returns_probabilities() -> None:
  # The TorchScript export returns logits; the backend applies the sigmoid the
  # other backends have baked in. Without it the values leave [0, 1].
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()

  model = load("geo", "3.0", "pt", precision="fp32")
  result = model.predict(20, 50, week=1, min_confidence=0.0, half_precision=False)

  assert result.species_probs.min() >= 0.0
  assert result.species_probs.max() <= 1.0
  # a range model that predicts nothing anywhere would also satisfy the bounds
  assert result.species_probs.max() > 0.5


def test_pt_matches_onnx() -> None:
  """The backends must agree - same species order, same probabilities.

  This is the assertion that catches a missing or misplaced sigmoid in the
  TorchScript path as well as a species list that drifted out of sync with the
  model outputs; the per-backend tests above pass in both cases.
  """
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()
  ensure_onnxruntime_or_skip()

  pt_model = load("geo", "3.0", "pt", precision="fp32")
  onnx_model = load("geo", "3.0", "onnx", precision="fp32")

  assert list(pt_model.species_list) == list(onnx_model.species_list)

  for latitude, longitude, week in ((42.5, -76.45, 4), (-33.9, 151.2, None)):
    pt_result = pt_model.predict(latitude, longitude, week=week, min_confidence=0.0)
    onnx_result = onnx_model.predict(latitude, longitude, week=week, min_confidence=0.0)
    np.testing.assert_allclose(
      pt_result.species_probs,
      onnx_result.species_probs,
      rtol=0,
      atol=1e-5,
      err_msg=f"pt and onnx disagree at ({latitude}, {longitude}, week={week})",
    )


@pytest.mark.gpu
def test_pt_fp32_gpu() -> None:
  ensure_gpu_or_skip()
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()

  model = load("geo", "3.0", "pt", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=False, device="GPU"
  )

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


# --- ONNX ---


def test_onnx_fp32() -> None:
  ensure_onnxruntime_or_skip()

  model = load("geo", "3.0", "onnx", precision="fp32")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_onnx_fp32_half() -> None:
  ensure_onnxruntime_or_skip()

  model = load("geo", "3.0", "onnx", precision="fp32")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=True)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_onnx_fp16() -> None:
  ensure_onnxruntime_or_skip()

  model = load("geo", "3.0", "onnx", precision="fp16")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp16"


def test_onnx_year_round() -> None:
  ensure_onnxruntime_or_skip()

  model = load("geo", "3.0", "onnx", precision="fp32")
  result = model.predict(20, 50, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == -1
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


@pytest.mark.gpu
def test_onnx_fp32_gpu() -> None:
  ensure_gpu_or_skip()
  ensure_onnxruntime_or_skip()

  model = load("geo", "3.0", "onnx", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=False, device="GPU"
  )

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"
