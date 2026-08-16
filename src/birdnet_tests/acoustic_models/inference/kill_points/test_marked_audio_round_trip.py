"""Every segment must land in its own row, not merely in some row.

The kill-point tests next door assert that a dying child ends the run. This
asserts the other half: that a *healthy* run puts each segment's result where
that segment belongs. Without it the seam could only ever count rows, and the
failure mode of a result path that reorders, drops or duplicates work is
silent — right numbers of rows, wrong contents.

That distinction is not hypothetical. It is exactly how the two designs
withdrawn from the crash-safe pipeline concept failed review: both produced
clean exit codes and missing segments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import (
  fake_encode_session,
  segment_marker,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_N_FILES = 3
_N_SEGMENTS = 6


@pytest.mark.parametrize("n_workers", [1, 3])
def test_each_segment_lands_in_its_own_row(tmp_path: Path, n_workers: int) -> None:
  """The marker written into segment i must come back in row i of file f.

  Several workers on purpose: blocks arrive interleaved from different
  processes, so this is the assertion that the ordering does not depend on
  which worker happened to finish first.
  """
  files = [
    str(write_marked_audio(tmp_path / f"in_{i}.wav", _N_SEGMENTS))
    for i in range(_N_FILES)
  ]

  with fake_encode_session(tmp_path, n_workers=n_workers, batch_size=2) as session:
    result = session.run(files)

  embeddings = result.embeddings
  assert embeddings.shape[0] == _N_FILES
  assert embeddings.shape[1] == _N_SEGMENTS

  expected = np.array([segment_marker(i) for i in range(_N_SEGMENTS)], dtype=np.float32)
  for file_idx in range(_N_FILES):
    actual = embeddings[file_idx, :, 0]
    np.testing.assert_allclose(
      actual,
      expected,
      rtol=0,
      atol=1e-6,
      err_msg=(
        f"file {file_idx}: segments came back in the wrong rows. "
        f"got {actual.tolist()}, want {expected.tolist()}"
      ),
    )


def test_the_marker_would_actually_catch_a_swap(tmp_path: Path) -> None:
  """Guards the guard: distinct markers, so a reordering is visible.

  If every segment carried the same value the test above would pass under any
  permutation, which is the decorative-test trap this whole file exists to
  avoid.
  """
  markers = [segment_marker(i) for i in range(_N_SEGMENTS)]
  assert len(set(markers)) == _N_SEGMENTS, "segment markers must be distinct"
