"""Healthy runs must survive result blocks big enough to tear.

The kill-point tests next door cover runs that *fail* at these block sizes;
nothing else runs one to completion. That gap mattered once already: the
readers that carry these blocks were rewired for issue #83, and only killed
runs and small-block healthy runs were covered by tests.

Block sizes here are past the 16 KB framing threshold (issue #83's surface):
a wide prediction block with ``top_k=None`` is ~45 KB, a batch-4 wide
encoding block ~17 KB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import (
  WideFakeAcousticBackend,
  fake_encode_session,
  fake_predict_session,
  segment_marker,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_N_FILES = 2
_N_SEGMENTS = 4


def _corpus(tmp_path: Path) -> list[str]:
  return [
    str(write_marked_audio(tmp_path / f"in_{i}.wav", _N_SEGMENTS))
    for i in range(_N_FILES)
  ]


def test_wide_prediction_completes_and_is_complete(tmp_path: Path) -> None:
  """Every segment present, across two runs of one session."""
  files = _corpus(tmp_path)
  with fake_predict_session(
    tmp_path, n_workers=2, top_k=None, backend=WideFakeAcousticBackend
  ) as session:
    first = session.run(files)
    second = session.run(files)

  for result in (first, second):
    assert result.species_probs.shape == (
      _N_FILES,
      _N_SEGMENTS,
      WideFakeAcousticBackend.n_species_out,
    )
    assert len(result.unprocessable_inputs) == 0


def test_wide_encoding_keeps_every_segment_in_its_own_row(tmp_path: Path) -> None:
  """The ordering assertion, at block sizes that cross the framing threshold.

  Same marker scheme as test_marked_audio_round_trip, so a result path that
  reorders or drops a block under multi-write frames fails on contents, not
  just on counts.
  """
  files = _corpus(tmp_path)
  with fake_encode_session(
    tmp_path, n_workers=2, batch_size=4, backend=WideFakeAcousticBackend
  ) as session:
    result = session.run(files)

  embeddings = result.embeddings
  assert embeddings.shape[:2] == (_N_FILES, _N_SEGMENTS)
  expected = np.array([segment_marker(i) for i in range(_N_SEGMENTS)], dtype=np.float32)
  for file_idx in range(_N_FILES):
    np.testing.assert_allclose(embeddings[file_idx, :, 0], expected, rtol=0, atol=1e-6)
