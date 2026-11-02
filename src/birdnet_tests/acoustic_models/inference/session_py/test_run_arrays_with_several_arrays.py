"""Several arrays in one ``run_arrays``: one row each, padded to the longest and masked.

The per-array shape is the session's, whatever the model; the rows are told
apart by the segment markers the fake backend hands back in the embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdnet_tests.fake_acoustic_backend import (
  DEFAULT_EMB_DIM,
  DEFAULT_N_SPECIES,
  fake_encode_session,
  fake_predict_session,
  segment_marker,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_SEGMENTS = (2, 3)


def _arrays(tmp_path: Path) -> list[tuple[np.ndarray, int]]:
  return [sf.read(write_marked_audio(tmp_path / f"{n}.wav", n)) for n in _SEGMENTS]


def _assert_only_the_padding_is_masked(masked: np.ndarray) -> None:
  assert not masked[0, :2].any()
  assert masked[0, 2].all()
  assert not masked[1].any()


def test_predict_gives_one_row_per_array(tmp_path: Path) -> None:
  with fake_predict_session(
    tmp_path, n_workers=1, top_k=None, default_confidence_threshold=-float("inf")
  ) as session:
    res = session.run_arrays(_arrays(tmp_path))

  assert res.species_probs.shape == (2, 3, DEFAULT_N_SPECIES)
  _assert_only_the_padding_is_masked(np.asarray(res.species_masked))


def test_encode_gives_one_row_per_array_with_its_markers(tmp_path: Path) -> None:
  with fake_encode_session(tmp_path, n_workers=1) as session:
    res = session.run_arrays(_arrays(tmp_path))

  embeddings = np.asarray(res.embeddings)
  assert embeddings.shape == (2, 3, DEFAULT_EMB_DIM)
  for i, n in enumerate(_SEGMENTS):
    expected = np.array([segment_marker(j) for j in range(n)], dtype=np.float32)
    np.testing.assert_array_equal(embeddings[i, :n, 0], expected, err_msg=f"array {i}")
  _assert_only_the_padding_is_masked(np.asarray(res.embeddings_masked))
