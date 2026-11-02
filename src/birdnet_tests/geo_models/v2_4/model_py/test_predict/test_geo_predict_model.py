import numpy as np
import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip

# What the result reports about the call (path, version, coordinates, week) is
# pinned on the fake in geo_models/base_py/test_geo_predict_forwards_its_arguments.py.
_CASES = [
  pytest.param("tflite", "CPU", id="tflite"),
  pytest.param("litert", "CPU", marks=pytest.mark.litert, id="litert"),
  pytest.param("pb", "CPU", id="pb-cpu"),
  pytest.param("pb", "GPU", marks=pytest.mark.gpu, id="pb-gpu"),
]


@pytest.mark.parametrize("half_precision", [False, True], ids=["fp32", "half"])
@pytest.mark.parametrize(("backend", "device"), _CASES)
def test_predict_shape_and_dtype(
  backend: str, device: str, half_precision: bool
) -> None:
  if backend == "litert":
    ensure_litert_or_skip()
  if device == "GPU":
    ensure_gpu_or_skip()
  if backend == "pb":
    model = load("geo", "2.4", "pb", precision="fp32")
  else:
    model = load("geo", "2.4", "tf", precision="fp32", library=backend)

  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=half_precision, device=device
  )

  assert result.species_probs.shape == (6522,)
  assert result.species_probs.dtype == (np.float16 if half_precision else np.float32)
  assert result.model_version == "2.4"
  assert result.model_precision == "fp32"
