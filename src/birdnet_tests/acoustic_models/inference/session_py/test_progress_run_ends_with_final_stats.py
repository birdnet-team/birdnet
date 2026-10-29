"""A run with progress tracking ends, and its last update says so.

The tracker keeps draining its stats queues after the session signals the
end of the run; a tracker that lost that signal between passes would report
forever and the session would wait on it forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf


def test_two_runs_each_end_with_a_finished_update(tmp_path: Path) -> None:
  updates: list[AcousticProgressStats] = []
  files = [str(write_marked_audio(tmp_path / f"in_{i}.wav", 6)) for i in range(2)]
  # Slow batches, so the run outlasts the tracker's first pass and the end
  # of the run is signalled while the tracker still has stats to drain.
  with fake_predict_session(
    tmp_path,
    n_workers=2,
    seconds_per_batch=0.25,
    show_stats="progress",
    progress_callback=updates.append,
  ) as session:
    for _ in range(2):
      n_before = len(updates)
      session.run(files)
      assert updates[-1].finished
      assert len(updates) > n_before
