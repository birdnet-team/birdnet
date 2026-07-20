"""Wiring test for ``on_file_complete`` on the BirdNET 3.0 predict wrapper.

The per-file callback behavior is covered in depth under ``v2_4``; here we only
assert the v3.0 model wrapper forwards the callback to the session.
"""

import threading
from typing import Literal

import pytest

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_onnxruntime_or_skip,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT

_Backend = Literal["pt", "onnx"]


def _load_model(backend: _Backend) -> AcousticModelV3_0:
  if backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
  else:
    ensure_onnxruntime_or_skip()

  return load("acoustic", "3.0", backend, precision="fp32")


class _Collector:
  """Thread-safe sink for per-file results (callbacks fire on a bg thread)."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self.results: dict[str, AcousticFilePredictionResult] = {}
    self.order: list[str] = []

  def __call__(self, result: AcousticFilePredictionResult) -> None:
    assert result.n_inputs == 1
    key = str(result.inputs[0])
    with self._lock:
      self.results[key] = result
      self.order.append(key)


@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_v3_0_predict_fires_once_per_file(backend: _Backend) -> None:
  collector = _Collector()
  model = _load_model(backend)
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  assert len(collector.order) == 2
  assert set(collector.results) == {
    str(TEST_FILE_SHORT.absolute()),
    str(TEST_FILE_LONG.absolute()),
  }
