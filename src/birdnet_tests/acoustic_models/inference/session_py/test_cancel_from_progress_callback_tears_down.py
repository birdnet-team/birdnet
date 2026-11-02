"""Cancelling a running session from its progress callback ends the run cleanly.

Cancelling mid-run (a GUI "stop" button wired to a progress callback, so the
call lands on the dispatcher thread) must end ``run()`` with a cancellation
error and let the ``with`` block exit. ``ProcessManager.join()`` is bounded by
``TEARDOWN_BUDGET_S``, so the deadline below catches a wedged parent, not a
slow teardown.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.acoustic.inference.session import AcousticSessionBase
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)
from birdnet_tests.helper import run_session_in_thread

pytestmark = pytest.mark.no_tf

# Bounds cancel-to-exit. A clean teardown takes a few seconds and the join at
# most TEARDOWN_BUDGET_S; only a wedged parent reaches this.
_TEARDOWN_DEADLINE_S = 60.0
# Sixteen segments at half a second over two workers is a four-second run, so
# the tracker's first update (after one second) lands with most of the work and
# its buffered result blocks still outstanding -- the state where teardown can
# deadlock. The run is cancelled right there, so the audio costs nothing.
_N_SEGMENTS_PER_FILE = 8
_STALL_S = 0.5


def test_cancel_from_progress_callback_tears_down_cleanly(tmp_path: Path) -> None:
  files = [
    str(write_marked_audio(tmp_path / f"in_{i}.wav", _N_SEGMENTS_PER_FILE))
    for i in range(2)
  ]
  holder: dict = {}
  cancelled = threading.Event()

  def on_progress(stats: AcousticProgressStats) -> None:
    # Only invoked once at least one prediction has been made, so the very
    # first call already means the run is under way with work remaining.
    session = holder.get("session")
    if session is not None and not cancelled.is_set():
      cancelled.set()
      # Mirrors cancelling from a GUI "stop" button; runs on the dispatcher thread.
      session.cancel()

  def remember(session: AcousticSessionBase) -> None:
    holder["session"] = session

  outcome = run_session_in_thread(
    lambda: fake_predict_session(
      tmp_path,
      n_workers=2,
      seconds_per_batch=_STALL_S,
      show_stats="progress",
      progress_callback=on_progress,
    ),
    files,
    remember,
    _TEARDOWN_DEADLINE_S,
  )

  assert outcome["finished"], (
    f"session did not tear down within {_TEARDOWN_DEADLINE_S:.0f} s after cancel()"
  )
  assert cancelled.is_set(), "the progress callback never ran, so nothing was cancelled"
  error = outcome.get("error")
  assert isinstance(error, RuntimeError) and "cancelled" in str(error), (
    f"cancel() must end run() with a cancellation; got {error!r}"
  )
