"""Regression tests for cancelling a running prediction session (issue #51).

Cancelling mid-run (e.g. a GUI "stop" button wired to a progress callback) must
raise inside ``run()`` and then let the ``with`` block exit promptly, instead of
hanging in ``ProcessManager.join()`` while it drains the worker queues.
"""

import threading

import pytest

from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG


def _load_model():  # noqa: ANN202
  return load("acoustic", "2.4", "tf", precision="fp32", library="tflite")


# Generous upper bound on cancel-to-teardown time: we cancel at ~10% progress, so
# only a little inference runs and a clean teardown finishes well under this. Kept
# below the global 300s per-test timeout, and enforced from a helper thread, so a
# regression fails the assertion fast instead of hanging and killing the worker.
_TEARDOWN_DEADLINE_S = 120.0


def test_cancel_from_progress_callback_tears_down_cleanly() -> None:
  model = _load_model()

  # Enough audio across multiple workers that the run lasts well beyond the first
  # progress callback, so cancelling then reliably lands mid-run with many
  # segments (and buffered result batches) still outstanding -- exactly the state
  # that used to deadlock teardown. The run is cancelled almost immediately, so
  # the large file list does not make the test slow.
  files = [str(TEST_FILE_LONG)] * 8

  holder: dict = {}
  cancelled = threading.Event()

  def on_progress(stats: AcousticProgressStats) -> None:
    # on_progress is only invoked once at least one prediction has been made, so
    # the very first call already means the run is under way with work remaining.
    session = holder.get("session")
    if session is not None and not cancelled.is_set():
      cancelled.set()
      # Mirrors cancelling from a GUI "stop" button; runs on the dispatcher thread.
      session.cancel()

  errors: list[BaseException] = []
  finished = threading.Event()

  def run_session() -> None:
    try:
      with model.predict_session(
        n_workers=2,
        top_k=5,
        show_stats="progress",
        progress_callback=on_progress,
      ) as session:
        holder["session"] = session
        with pytest.raises(RuntimeError, match="cancelled"):
          session.run(files)
      # Reaching here means __exit__ (join + teardown) returned without hanging.
    except BaseException as exc:  # noqa: BLE001 - surfaced via the assertions below
      errors.append(exc)
    finally:
      finished.set()

  worker = threading.Thread(
    target=run_session, name="cancel-teardown-test", daemon=True
  )
  worker.start()

  completed = finished.wait(timeout=_TEARDOWN_DEADLINE_S)
  assert completed, (
    f"session did not tear down within {_TEARDOWN_DEADLINE_S:.0f}s after "
    f"cancel() -- ProcessManager.join() likely hung"
  )
  assert cancelled.is_set(), "progress callback never reached the cancel threshold"
  assert not errors, f"unexpected error during cancelled run/teardown: {errors!r}"
