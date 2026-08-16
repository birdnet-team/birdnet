"""A run that predicts nothing must still finish, and still call back.

`ProgressDispatcher` stops when it sees stats flagged ``finished``. The
performance tracker used to skip publishing stats entirely while no worker had
reported a timing, so a run where every input is unprocessable produced none at
all: the dispatcher looped forever, never set its finish signal, and the parent
waited on that signal for good (issue #75).

The hang is in a background thread, so the whole session runs in a worker
thread here with a deadline -- otherwise a regression would wedge the test
until the suite-level timeout instead of failing with a readable message.
"""

import threading
from pathlib import Path

from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_SHORT

# A healthy empty run takes a couple of seconds; the guarded regression hangs.
_DEADLINE_S = 90.0
# Enough reuse for the dispatcher to fall a run behind if it can.
_N_RUNS = 3


def _broken_wav(tmp_path: Path) -> str:
  target = tmp_path / "broken.wav"
  target.write_bytes(b"NOT_A_VALID_WAV_FILE")
  return str(target)


def test_run_without_predictions_finishes_and_reports_finished(tmp_path: Path) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  broken = _broken_wav(tmp_path)

  seen: list[AcousticProgressStats] = []
  outcome: dict = {}
  done = threading.Event()

  def on_progress(stats: AcousticProgressStats) -> None:
    seen.append(stats)

  def run() -> None:
    try:
      with model.predict_session(
        n_workers=1,
        top_k=None,
        show_stats="progress",
        progress_callback=on_progress,
      ) as session:
        outcome["result"] = session.run([broken])
    except BaseException as e:  # noqa: BLE001 - surfaced by the assertions below
      outcome["error"] = e
    finally:
      done.set()

  runner = threading.Thread(target=run, name="empty-run", daemon=True)
  runner.start()

  assert done.wait(timeout=_DEADLINE_S), (
    f"the run did not finish within {_DEADLINE_S:.0f} s; the progress "
    f"dispatcher is still waiting for stats that will never be published"
  )
  assert "error" not in outcome, f"the run failed: {outcome['error']!r}"

  result = outcome["result"]
  assert result.get_unprocessed_files() == {Path(broken).absolute()}

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
  """Every run must get its closing update, and the session must still close.

  The parent only waits for the dispatcher if its finish signal was cleared
  between runs. While it was left set, runs after the first raced ahead of the
  dispatcher: they got no callback at all, and the dispatcher drifted a run
  behind until nothing could end it and closing the session blocked forever.
  Both failures are invisible from a single run, which is why this reuses one.
  """
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  target = str(TEST_FILE_SHORT)

  per_run: list[int] = []
  seen: list[AcousticProgressStats] = []
  outcome: dict = {}
  done = threading.Event()

  def on_progress(stats: AcousticProgressStats) -> None:
    seen.append(stats)

  def run() -> None:
    try:
      with model.predict_session(
        n_workers=1,
        top_k=None,
        show_stats="progress",
        progress_callback=on_progress,
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

  runner = threading.Thread(target=run, name="reused-session", daemon=True)
  runner.start()

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
