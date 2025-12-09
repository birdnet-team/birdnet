import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip


@pytest.mark.litert
def test_litert() -> None:
  ensure_litert_or_skip()

  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


def test_tf() -> None:
  model = load("geo", "2.4", "tf", precision="fp32", library="tflite")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


def test_pb_cpu() -> None:
  model = load("geo", "2.4", "pb", precision="fp32")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=False)

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


@pytest.mark.gpu
def test_pb_gpu() -> None:
  ensure_gpu_or_skip()

  model = load("geo", "2.4", "pb", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=False, device="GPU"
  )

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


@pytest.mark.litert
def test_litert_half() -> None:
  ensure_litert_or_skip()

  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=True)

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


def test_tf_half() -> None:
  model = load("geo", "2.4", "tf", precision="fp32", library="tflite")
  result = model.predict(20, 50, week=1, min_confidence=0.03, half_precision=True)

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


def test_pb_cpu_half() -> None:
  model = load("geo", "2.4", "pb", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=True, device="CPU"
  )

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"


@pytest.mark.gpu
def test_pb_gpu_half() -> None:
  ensure_gpu_or_skip()

  model = load("geo", "2.4", "pb", precision="fp32")
  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=True, device="GPU"
  )

  assert result.species_probs.shape == (6522,)
  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == 1
  assert result.model_path == model.model_path.absolute()
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"
