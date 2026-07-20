"""Tests for the per-file completion callback (``on_file_complete``)."""

import tempfile
import threading
from pathlib import Path

import numpy
import pytest

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.model_loader import load
from birdnet_tests.helper import create_zero_len_wav
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT


def _load_model():  # noqa: ANN202
  return load("acoustic", "2.4", "tf", precision="fp32", library="tflite")


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


def test_fires_once_per_file() -> None:
  collector = _Collector()
  model = _load_model()
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  assert len(collector.order) == 2
  assert set(collector.results) == {
    str(TEST_FILE_SHORT.absolute()),
    str(TEST_FILE_LONG.absolute()),
  }


def test_per_file_result_matches_aggregate() -> None:
  collector = _Collector()
  model = _load_model()
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    aggregate = session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  aggregate_inputs = list(aggregate.inputs)
  for key, per_file in collector.results.items():
    idx = aggregate_inputs.index(key)
    n_seg = per_file.species_probs.shape[1]
    numpy.testing.assert_array_equal(
      per_file.species_probs[0], aggregate.species_probs[idx, :n_seg]
    )
    numpy.testing.assert_array_equal(
      per_file.species_ids[0], aggregate.species_ids[idx, :n_seg]
    )
    numpy.testing.assert_array_equal(
      per_file.species_masked[0], aggregate.species_masked[idx, :n_seg]
    )


def test_per_file_segment_counts() -> None:
  collector = _Collector()
  model = _load_model()
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  short = collector.results[str(TEST_FILE_SHORT.absolute())]
  long = collector.results[str(TEST_FILE_LONG.absolute())]
  assert short.species_probs.shape == (1, 3, 5)
  assert long.species_probs.shape == (1, 40, 5)


def test_per_file_result_to_dataframe() -> None:
  collector = _Collector()
  model = _load_model()
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run(str(TEST_FILE_SHORT))

  result = collector.results[str(TEST_FILE_SHORT.absolute())]
  df = result.to_dataframe()
  assert len(df) > 0
  assert set(df.columns) >= {"input", "start_time", "end_time"}


def test_invalid_file_reported_as_unprocessable() -> None:
  collector = _Collector()
  model = _load_model()
  with tempfile.NamedTemporaryFile(
    suffix=".wav", delete=False, mode="wb"
  ) as broken:
    broken.write(b"NOT_A_VALID_WAV_FILE")
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run([broken.name, str(TEST_FILE_SHORT)])
  broken.close()

  assert len(collector.order) == 2
  invalid_result = collector.results[str(Path(broken.name).absolute())]
  assert invalid_result.get_unprocessed_files() == {Path(broken.name).absolute()}
  assert invalid_result.species_probs.shape[1] == 0

  valid_result = collector.results[str(TEST_FILE_SHORT.absolute())]
  assert valid_result.get_unprocessed_files() == set()
  assert valid_result.species_probs.shape[1] == 3


def test_empty_but_readable_file_is_zero_detection_not_unprocessable() -> None:
  collector = _Collector()
  model = _load_model()
  with tempfile.NamedTemporaryFile(
    suffix=".wav", delete=False, mode="wb"
  ) as empty_wav:
    create_zero_len_wav(empty_wav)
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run(empty_wav.name)
  empty_wav.close()

  result = collector.results[str(Path(empty_wav.name).absolute())]
  assert result.get_unprocessed_files() == set()
  assert result.species_probs.shape == (1, 0, 5)


def test_two_producers_fire_each_file_once() -> None:
  collector = _Collector()
  model = _load_model()
  with model.predict_session(
    n_workers=1, n_producers=2, top_k=5, on_file_complete=collector
  ) as session:
    session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  assert sorted(collector.order) == sorted(
    [str(TEST_FILE_SHORT.absolute()), str(TEST_FILE_LONG.absolute())]
  )


def test_same_session_twice_resets_between_runs() -> None:
  collector = _Collector()
  model = _load_model()
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    session.run(str(TEST_FILE_SHORT))
    assert len(collector.order) == 1
    session.run(str(TEST_FILE_LONG))

  assert len(collector.order) == 2


def test_run_arrays_with_callback_raises() -> None:
  import soundfile as sf

  collector = _Collector()
  model = _load_model()
  sf_read = sf.read(TEST_FILE_SHORT)
  with model.predict_session(
    n_workers=1, top_k=5, on_file_complete=collector
  ) as session:
    with pytest.raises(RuntimeError, match="only supported for file inputs"):
      session.run_arrays(sf_read)


def test_raising_callback_cancels_run() -> None:
  def boom(_result: AcousticFilePredictionResult) -> None:
    raise ValueError("callback failure")

  model = _load_model()
  with pytest.raises(RuntimeError):
    with model.predict_session(
      n_workers=1, top_k=5, on_file_complete=boom
    ) as session:
      session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])


def test_non_callable_callback_rejected() -> None:
  model = _load_model()
  with pytest.raises(TypeError, match="callable"):
    model.predict_session(n_workers=1, top_k=5, on_file_complete=123)
