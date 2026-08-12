"""A worker dying mid-run must fail the run, not hang it forever.

The parent waits for one sentinel per worker on the results queue and for each
child's finish signal, both without a timeout, because a healthy run takes as
long as the audio requires. A worker killed from the outside -- the OOM killer
under memory pressure, or a native crash -- never sends either, so before the
liveness check the parent waited forever with no output at all: the user saw
birdnet freeze with no way to tell a dead worker from a working one.

Killing the worker with SIGKILL is exactly what the OOM killer does, so this
reproduces the real failure rather than a simulation of it.
"""

import shutil
import threading
from pathlib import Path

import pytest

from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG

# Generous upper bound: without the liveness check this test hangs until the
# suite-level timeout kills the process. The check reports within one poll
# interval, so anything in this range means "failed instead of hanging".
_FAILURE_DEADLINE_S = 120.0


def test_killed_worker_fails_the_run_instead_of_hanging(tmp_path: Path) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")

  # Enough audio that the run is still going when the first progress callback
  # arrives, so the worker is killed mid-run with work outstanding. Distinct
  # copies because validate_input_files de-duplicates by absolute path.
  src = Path(TEST_FILE_LONG)
  files = [
    str(shutil.copyfile(src, tmp_path / f"copy_{i}{src.suffix}")) for i in range(4)
  ]

  holder: dict = {}
  killed = threading.Event()

  def on_progress(stats: AcousticProgressStats) -> None:
    # Fires once at least one prediction was made, so the pipeline is up and
    # the worker is doing real work.
    manager = holder.get("manager")
    if manager is None or killed.is_set():
      return
    workers = manager._worker_processes
    if not workers:
      return
    killed.set()
    workers[0].kill()

  with model.predict_session(
    n_workers=1,
    top_k=None,
    show_stats="progress",
    progress_callback=on_progress,
  ) as session:
    holder["manager"] = session._process_manager

    finished = threading.Event()
    outcome: dict = {}

    def run() -> None:
      try:
        session.run(files)
      except BaseException as e:  # noqa: BLE001 - recorded, re-checked below
        outcome["error"] = e
      finally:
        finished.set()

    runner = threading.Thread(target=run, name="dead-worker-run", daemon=True)
    runner.start()

    assert finished.wait(timeout=_FAILURE_DEADLINE_S), (
      f"run() did not return within {_FAILURE_DEADLINE_S:.0f} s after the worker "
      f"was killed; the parent is waiting on a child that no longer exists."
    )

  assert killed.is_set(), "the worker was never killed, so nothing was tested"
  assert "error" in outcome, "run() returned successfully despite a killed worker"
  assert isinstance(outcome["error"], RuntimeError), (
    f"expected the run to be reported as cancelled, got {outcome['error']!r}"
  )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_healthy_run_is_unaffected(tmp_path: Path) -> None:
  """The liveness check must not disturb a run where nothing dies."""
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  src = Path(TEST_FILE_LONG)
  target = str(shutil.copyfile(src, tmp_path / f"copy{src.suffix}"))

  with model.predict_session(n_workers=1, top_k=None) as session:
    result = session.run([target])

  assert result is not None
