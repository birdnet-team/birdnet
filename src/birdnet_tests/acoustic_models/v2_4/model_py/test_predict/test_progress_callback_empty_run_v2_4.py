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

# A healthy empty run takes a couple of seconds; without the fix it never ends.
_DEADLINE_S = 90.0


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

  # The callback contract is that the final stats always arrive. Before the fix
  # this run produced no callback at all, so asserting only that the run
  # completes would miss half the defect.
  assert seen, "the progress callback was never called"
  assert seen[-1].finished, (
    f"the last stats were not flagged finished: {seen[-1].finished}"
  )
  assert seen[-1].processed_segments == 0
  assert seen[-1].worker_stats is None, (
    "no worker ran, so there are no worker stats to report"
  )
