"""A run whose performance tracker process was killed mid-run fails with a diagnosis.

The tracker sleeps on the processing-finished signal from a few milliseconds
into a run until the parent announces the end of the run. That signal is a
semaphore because a release returns even when the sleeper was killed inside
its wait; `multiprocessing.Event.set()` waits for every registered waiter to
confirm it woke, forever when that waiter is dead, and the parent reaches it
before any liveness check can notice the death. The tracker is killed
between the consumer's return and that announcement, where no liveness check
runs. The run and the exit each execute on a daemon thread so that a hang
fails the test instead of wedging it; on failure the session's remaining
children are killed rather than joined.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_DEADLINE_S = 20.0


def _within_deadline(step: Callable[[], object], name: str) -> BaseException | None:
  done = threading.Event()
  outcome: dict[str, BaseException] = {}

  def call() -> None:
    try:
      step()
    except BaseException as e:  # noqa: BLE001
      outcome["error"] = e
    finally:
      done.set()

  threading.Thread(target=call, name=name, daemon=True).start()
  assert done.wait(_DEADLINE_S), (
    f"{name} did not return within {_DEADLINE_S:.0f} s after the performance "
    f"tracker had died; it is wedged on a child that no longer exists"
  )
  return outcome.get("error")


def test_run_raises_instead_of_hanging(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # Long enough for the tracker to be well inside its first one-second wait
  # when the consumer returns: 12 single-segment batches of 50 ms on 2 workers.
  audio = write_marked_audio(tmp_path / "in.wav", 12)
  session = fake_predict_session(
    tmp_path, n_workers=2, show_stats="progress", seconds_per_batch=0.05
  )
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  tracker = manager._perf_tracker_process  # noqa: SLF001
  assert tracker is not None
  children = [
    *(manager._producer_processes or []),  # noqa: SLF001
    *(manager._worker_processes or []),  # noqa: SLF001
    tracker,
  ]
  run_consumer = manager.run_consumer

  def run_consumer_then_kill_tracker(*args: object, **kwargs: object) -> None:
    run_consumer(*args, **kwargs)  # type: ignore[arg-type]
    tracker.kill()
    tracker.join()

  monkeypatch.setattr(manager, "run_consumer", run_consumer_then_kill_tracker)

  left = False
  try:
    error = _within_deadline(lambda: session.run([str(audio)]), "run()")
    assert isinstance(error, RuntimeError), error
    assert "Pipeline process" in str(error)

    assert (
      _within_deadline(lambda: session.__exit__(None, None, None), "__exit__") is None
    )
    left = True
  finally:
    if not left:
      for child in children:
        child.kill()
