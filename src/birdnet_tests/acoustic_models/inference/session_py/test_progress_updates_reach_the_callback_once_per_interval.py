"""A run's progress callback hears from the tracker about once per interval.

The tracker sends an update every interval (1 s) with the stats collected at
that moment, and the dispatcher hands each to the callback as it arrives, so
the callback runs once per interval, with the closing update after the last
ordinary one.

- The call count is a coarse floor. A tracker that collects before its wait,
  paired with a dispatcher that holds each update for a quiet interval, made 3
  calls on this run in most tries but not all. That dispatcher alone made 4 or
  5; its own unit test is what catches it.
- The closing update at the full count is a guard here, not a proof: a close
  that misses stats trailing the end-of-run token passed this test in every
  run tried; the tracker's own unit test enforces it.
- Progress never going backwards is an invariant, not tied to a mutation: the
  tracker's count only grows and its updates arrive in order.
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

# 40 single-segment batches of 250 ms on 2 workers keep the run going for at
# least 4.75 s after the first batch is in: four interval updates and the
# closing one, six calls in practice. The batches sleep, so load does not
# stretch the run; it slows the tracker's passes, which removes updates. Four
# leaves two lost updates of margin.
_N_SEGMENTS = 40
_MIN_CALLS = 4


def test_a_five_second_run_gets_an_update_every_interval(tmp_path: Path) -> None:
  updates: list[AcousticProgressStats] = []
  audio = write_marked_audio(tmp_path / "in.wav", _N_SEGMENTS)
  with fake_predict_session(
    tmp_path,
    n_workers=2,
    seconds_per_batch=0.25,
    show_stats="progress",
    progress_callback=updates.append,
  ) as session:
    session.run([str(audio)])

  assert len(updates) >= _MIN_CALLS, (
    f"the callback ran {len(updates)} times in a run of at least 5 s, "
    f"expected at least {_MIN_CALLS}"
  )
  processed = [update.processed_segments for update in updates]
  assert processed == sorted(processed), f"progress went backwards: {processed}"
  assert not any(update.finished for update in updates[:-1])
  last = updates[-1]
  assert last.finished
  assert last.processed_segments == _N_SEGMENTS
  assert last.progress_pct == 100.0
