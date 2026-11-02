"""The per-file completion callback (``on_file_complete``) on encoding runs.

Nothing asserted here depends on the model, so the pipeline runs on the fake
backend. One run on the real V2.4 model pins the wiring in
``acoustic_models/v2_4/model_py/test_encode/test_on_file_complete_encode_v2_4.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet_tests.fake_acoustic_backend import (
  DEFAULT_EMB_DIM,
  fake_encode_session,
  write_marked_audio,
)
from birdnet_tests.helper import FileResultCollector

pytestmark = pytest.mark.no_tf

_SHORT_SEGMENTS = 2
_LONG_SEGMENTS = 4


def _collector() -> FileResultCollector[AcousticFileEncodingResult]:
  return FileResultCollector()


def _short_file(tmp_path: Path) -> Path:
  return write_marked_audio(tmp_path / "short.wav", _SHORT_SEGMENTS)


def _long_file(tmp_path: Path) -> Path:
  return write_marked_audio(tmp_path / "long.wav", _LONG_SEGMENTS)


def test_fires_once_per_file(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_encode_session(tmp_path, n_workers=1, on_file_complete=collector) as s:
    s.run([str(short), str(long)])

  assert len(collector.order) == 2
  assert set(collector.results) == {str(short.absolute()), str(long.absolute())}


def test_per_file_embeddings_match_aggregate(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_encode_session(tmp_path, n_workers=1, on_file_complete=collector) as s:
    aggregate = s.run([str(short), str(long)])

  # Said before the loop: over an empty collector the loop would pass vacuously.
  assert set(collector.results) == set(aggregate.inputs)
  aggregate_inputs = list(aggregate.inputs)
  for key, per_file in collector.results.items():
    idx = aggregate_inputs.index(key)
    n_seg = per_file.embeddings.shape[1]
    np.testing.assert_array_equal(
      per_file.embeddings[0], aggregate.embeddings[idx, :n_seg]
    )
    np.testing.assert_array_equal(
      per_file.embeddings_masked[0], aggregate.embeddings_masked[idx, :n_seg]
    )


def test_per_file_segment_counts(tmp_path: Path) -> None:
  short, long = _short_file(tmp_path), _long_file(tmp_path)
  collector = _collector()
  with fake_encode_session(tmp_path, n_workers=1, on_file_complete=collector) as s:
    s.run([str(short), str(long)])

  short_result = collector.results[str(short.absolute())]
  long_result = collector.results[str(long.absolute())]
  assert short_result.embeddings.shape == (1, _SHORT_SEGMENTS, DEFAULT_EMB_DIM)
  assert long_result.embeddings.shape == (1, _LONG_SEGMENTS, DEFAULT_EMB_DIM)


def test_invalid_file_reported_as_unprocessable(tmp_path: Path) -> None:
  short = _short_file(tmp_path)
  broken = tmp_path / "broken.wav"
  broken.write_bytes(b"NOT_A_VALID_WAV_FILE")
  collector = _collector()
  with fake_encode_session(tmp_path, n_workers=1, on_file_complete=collector) as s:
    s.run([str(broken), str(short)])

  assert len(collector.order) == 2
  invalid_result = collector.results[str(broken.absolute())]
  assert list(invalid_result.unprocessable_inputs()) == [0]
  assert invalid_result.embeddings.shape[1] == 0

  valid_result = collector.results[str(short.absolute())]
  assert list(valid_result.unprocessable_inputs()) == []
  assert valid_result.embeddings.shape[1] == _SHORT_SEGMENTS


def test_run_arrays_with_callback_raises(tmp_path: Path) -> None:
  short = _short_file(tmp_path)
  collector = _collector()
  with (
    fake_encode_session(tmp_path, n_workers=1, on_file_complete=collector) as s,
    pytest.raises(RuntimeError, match="only supported for file inputs"),
  ):
    s.run_arrays(sf.read(short))
