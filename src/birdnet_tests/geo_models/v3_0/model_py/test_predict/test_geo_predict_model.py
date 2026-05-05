import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip


# --- FP32 ---


@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="fp32", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_tf_fp32() -> None:
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
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="fp32", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=True)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


def test_tf_fp32_half() -> None:
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
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="fp16", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "fp16"


def test_tf_fp16() -> None:
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
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="int8", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "3.0"
  assert result.model_precision == "int8"


def test_tf_int8() -> None:
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
