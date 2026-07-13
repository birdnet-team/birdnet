"""Tests for the per-file completion callback (``on_file_complete``) in encoding."""

import tempfile
import threading
from pathlib import Path

import numpy
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT


def _load_model():  # noqa: ANN202
  return load("acoustic", "2.4", "tf", precision="fp32", library="tflite")


class _Collector:
  """Thread-safe sink for per-file results (callbacks fire on a bg thread)."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self.results: dict[str, AcousticFileEncodingResult] = {}
    self.order: list[str] = []

  def __call__(self, result: AcousticFileEncodingResult) -> None:
    assert result.n_inputs == 1
    key = str(result.inputs[0])
    with self._lock:
      self.results[key] = result
      self.order.append(key)


def test_fires_once_per_file() -> None:
  collector = _Collector()
  model = _load_model()
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  assert len(collector.order) == 2
  assert set(collector.results) == {
    str(TEST_FILE_SHORT.absolute()),
    str(TEST_FILE_LONG.absolute()),
  }


def test_per_file_embeddings_match_aggregate() -> None:
  collector = _Collector()
  model = _load_model()
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    aggregate = session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  aggregate_inputs = list(aggregate.inputs)
  for key, per_file in collector.results.items():
    idx = aggregate_inputs.index(key)
    n_seg = per_file.embeddings.shape[1]
    numpy.testing.assert_array_equal(
      per_file.embeddings[0], aggregate.embeddings[idx, :n_seg]
    )
    numpy.testing.assert_array_equal(
      per_file.embeddings_masked[0], aggregate.embeddings_masked[idx, :n_seg]
    )


def test_per_file_segment_counts() -> None:
  collector = _Collector()
  model = _load_model()
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  short = collector.results[str(TEST_FILE_SHORT.absolute())]
  long = collector.results[str(TEST_FILE_LONG.absolute())]
  assert short.embeddings.shape == (1, 3, 1024)
  assert long.embeddings.shape == (1, 40, 1024)


def test_invalid_file_reported_as_unprocessable() -> None:
  collector = _Collector()
  model = _load_model()
  with tempfile.NamedTemporaryFile(
    suffix=".wav", delete=False, mode="wb"
  ) as broken:
    broken.write(b"NOT_A_VALID_WAV_FILE")
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    session.run([broken.name, str(TEST_FILE_SHORT)])
  broken.close()

  assert len(collector.order) == 2
  invalid_result = collector.results[str(Path(broken.name).absolute())]
  assert list(invalid_result.unprocessable_inputs()) == [0]
  assert invalid_result.embeddings.shape[1] == 0

  valid_result = collector.results[str(TEST_FILE_SHORT.absolute())]
  assert list(valid_result.unprocessable_inputs()) == []
  assert valid_result.embeddings.shape[1] == 3


def test_run_arrays_with_callback_raises() -> None:
  import soundfile as sf

  collector = _Collector()
  model = _load_model()
  sf_read = sf.read(TEST_FILE_SHORT)
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    with pytest.raises(RuntimeError, match="only supported for file inputs"):
      session.run_arrays(sf_read)


def test_non_callable_callback_rejected() -> None:
  model = _load_model()
  with pytest.raises(TypeError, match="callable"):
    model.encode_session(n_workers=1, on_file_complete=123)
