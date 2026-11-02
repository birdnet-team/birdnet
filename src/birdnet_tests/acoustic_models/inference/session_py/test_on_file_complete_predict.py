"""The per-file completion callback (``on_file_complete``) on prediction runs.

Nothing asserted here depends on the model, so the pipeline runs on the fake
backend. One run on the real V2.4 model pins the wiring in
``acoustic_models/v2_4/model_py/test_predict/test_on_file_complete_v2_4.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)
from birdnet_tests.helper import FileResultCollector, create_zero_len_wav

pytestmark = pytest.mark.no_tf

_TOP_K = 5
_SHORT_SEGMENTS = 2
_LONG_SEGMENTS = 4


def _collector() -> FileResultCollector[AcousticFilePredictionResult]:
  return FileResultCollector()


def _short_file(tmp_path: Path) -> Path:
  return write_marked_audio(tmp_path / "short.wav", _SHORT_SEGMENTS)


def _long_file(tmp_path: Path) -> Path:
  return write_marked_audio(tmp_path / "long.wav", _LONG_SEGMENTS)


def test_fires_once_per_file(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run([str(short), str(long)])

  assert len(collector.order) == 2
  assert set(collector.results) == {str(short.absolute()), str(long.absolute())}


def test_per_file_result_matches_aggregate(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    aggregate = session.run([str(short), str(long)])

  # Said before the loop: over an empty collector the loop would pass vacuously.
  assert set(collector.results) == set(aggregate.inputs)
  aggregate_inputs = list(aggregate.inputs)
  for key, per_file in collector.results.items():
    idx = aggregate_inputs.index(key)
    n_seg = per_file.species_probs.shape[1]
    np.testing.assert_array_equal(
      per_file.species_probs[0], aggregate.species_probs[idx, :n_seg]
    )
    np.testing.assert_array_equal(
      per_file.species_ids[0], aggregate.species_ids[idx, :n_seg]
    )
    np.testing.assert_array_equal(
      per_file.species_masked[0], aggregate.species_masked[idx, :n_seg]
    )


def test_per_file_segment_counts(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run([str(short), str(long)])

  short_result = collector.results[str(short.absolute())]
  long_result = collector.results[str(long.absolute())]
  assert short_result.species_probs.shape == (1, _SHORT_SEGMENTS, _TOP_K)
  assert long_result.species_probs.shape == (1, _LONG_SEGMENTS, _TOP_K)


def test_per_file_result_to_dataframe(tmp_path: Path) -> None:
  short = _short_file(tmp_path)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run(str(short))

  df = collector.results[str(short.absolute())].to_dataframe()
  assert len(df) > 0
  assert set(df.columns) >= {"input", "start_time", "end_time"}


def test_invalid_file_reported_as_unprocessable(tmp_path: Path) -> None:
  short = _short_file(tmp_path)
  broken = tmp_path / "broken.wav"
  broken.write_bytes(b"NOT_A_VALID_WAV_FILE")
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run([str(broken), str(short)])

  assert len(collector.order) == 2
  invalid_result = collector.results[str(broken.absolute())]
  assert invalid_result.get_unprocessed_files() == {broken.absolute()}
  assert invalid_result.species_probs.shape[1] == 0

  valid_result = collector.results[str(short.absolute())]
  assert valid_result.get_unprocessed_files() == set()
  assert valid_result.species_probs.shape[1] == _SHORT_SEGMENTS


def test_empty_but_readable_file_is_zero_detection_not_unprocessable(
  tmp_path: Path,
) -> None:
  empty = tmp_path / "empty.wav"
  with empty.open("wb") as f:
    create_zero_len_wav(f)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run(str(empty))

  result = collector.results[str(empty.absolute())]
  assert result.get_unprocessed_files() == set()
  assert result.species_probs.shape == (1, 0, _TOP_K)


def test_two_producers_fire_each_file_once(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, n_producers=2, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run([str(short), str(long)])

  assert sorted(collector.order) == sorted(
    [str(short.absolute()), str(long.absolute())]
  )


def test_same_session_twice_resets_between_runs(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
  ) as session:
    session.run(str(short))
    assert len(collector.order) == 1
    session.run(str(long))

  assert len(collector.order) == 2


def test_run_arrays_with_callback_raises(tmp_path: Path) -> None:
  short = _short_file(tmp_path)
  collector = _collector()
  with (
    fake_predict_session(
      tmp_path, n_workers=1, top_k=_TOP_K, on_file_complete=collector
    ) as session,
    pytest.raises(RuntimeError, match="only supported for file inputs"),
  ):
    session.run_arrays(sf.read(short))
