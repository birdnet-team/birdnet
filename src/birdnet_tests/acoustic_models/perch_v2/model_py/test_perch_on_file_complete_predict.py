"""Wiring test for ``on_file_complete`` on the Perch V2 predict wrapper.

The per-file callback behavior is covered in depth under ``v2_4``; here we only
assert the Perch V2 model wrapper forwards the callback to the session.
"""

import threading

import pytest

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import (
  ensure_not_intel_macos_or_skip,
  ensure_tf_2_20_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT


@pytest.fixture(autouse=True)
def _ensure_supported_tf() -> None:
  ensure_tf_2_20_or_skip()


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


def test_perch_v2_predict_fires_once_per_file() -> None:
  ensure_not_intel_macos_or_skip()

  collector = _Collector()
  model = load_perch_v2("CPU")
  with model.predict_session(
    n_workers=1, top_k=None, device="CPU", on_file_complete=collector
  ) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  assert len(collector.order) == 2
  assert set(collector.results) == {
    str(TEST_FILE_SHORT.absolute()),
    str(TEST_FILE_LONG.absolute()),
  }
