"""Every run gets its closing progress update, even one that predicts nothing.

`ProgressDispatcher` stops when it sees stats flagged ``finished``, so the
tracker must publish a finished update even when no worker ever reported a
timing: a run where every input is unprocessable otherwise produces no update
at all, the dispatcher never sets its finish signal, and the parent waits on it
for good (issue #75). The reuse test pins that every run of a reused session
gets its own callbacks and closing update; the per-run clearing of the signals
that ordering rests on is pinned directly in
``resources_py/StatisticsResources/test_reset_clears_the_run_signals.py``.

A regression here hangs a background thread, so each session runs in a worker
thread with a deadline and fails with a readable message instead of wedging
the test until the suite-level timeout.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)
from birdnet_tests.helper import run_session_in_thread

pytestmark = pytest.mark.no_tf

# A healthy run here takes well under a second; the guarded regression hangs,
# or fails at the pipeline's own 30 s deadline for a promised message.
_DEADLINE_S = 90.0
# Enough reuse for the dispatcher to fall a run behind if it can.
_N_RUNS = 3


def test_run_without_predictions_finishes_and_reports_finished(
  tmp_path: Path,
) -> None:
  broken = tmp_path / "broken.wav"
  broken.write_bytes(b"NOT_A_VALID_WAV_FILE")
  seen: list[AcousticProgressStats] = []

  outcome = run_session_in_thread(
    lambda: fake_predict_session(
      tmp_path,
      n_workers=1,
      top_k=None,
      show_stats="progress",
      progress_callback=seen.append,
    ),
    [str(broken)],
    lambda session: None,
    _DEADLINE_S,
  )

  assert outcome["finished"], (
    f"the run did not finish within {_DEADLINE_S:.0f} s; the progress "
    f"dispatcher is still waiting for stats that will never be published"
  )
  assert "error" not in outcome, f"the run failed: {outcome['error']!r}"
  assert outcome["result"].get_unprocessed_files() == {broken.absolute()}

  # The callback contract is that the final stats always arrive - a regression
  # can complete the run yet publish no callback at all.
  assert seen, "the progress callback was never called"
  assert seen[-1].finished, (
    f"the last stats were not flagged finished: {seen[-1].finished}"
  )
  assert seen[-1].processed_segments == 0
  assert seen[-1].worker_stats is None, (
    "nothing was inferred, so there are no worker stats to report"
  )


def test_reused_session_keeps_reporting_and_closes(tmp_path: Path) -> None:
  target = str(write_marked_audio(tmp_path / "in.wav", 2))
  per_run: list[int] = []
  seen: list[AcousticProgressStats] = []
  outcome: dict = {}
  done = threading.Event()

  def run() -> None:
    try:
      with fake_predict_session(
        tmp_path,
        n_workers=1,
        top_k=None,
        show_stats="progress",
        progress_callback=seen.append,
      ) as session:
        for _ in range(_N_RUNS):
          before = len(seen)
          session.run([target])
          per_run.append(len(seen) - before)
      # Reaching here means the `with` block closed, which is half the point.
      outcome["closed"] = True
    except BaseException as e:  # noqa: BLE001 - surfaced by the assertions below
      outcome["error"] = e
    finally:
      done.set()

  threading.Thread(target=run, name="reused-session", daemon=True).start()

  assert done.wait(timeout=_DEADLINE_S), (
    f"{_N_RUNS} runs plus closing the session did not finish within "
    f"{_DEADLINE_S:.0f} s; the progress dispatcher is out of step with the runs"
  )
  assert "error" not in outcome, f"the session failed: {outcome['error']!r}"
  assert outcome.get("closed"), "the session never closed"

  assert all(n > 0 for n in per_run), (
    f"a run produced no progress callback at all: {per_run}"
  )
  assert sum(1 for s in seen if s.finished) == _N_RUNS, (
    f"expected one closing update per run, got "
    f"{sum(1 for s in seen if s.finished)} over {_N_RUNS} runs"
  )
