"""A session reused after an idle worker died fails that run with a diagnosis.

Starting a run releases every child's start signal. With a semaphore the
release returns even when the child was killed inside its wait, so the run
proceeds and the liveness check reports the dead process; with an Event the
parent blocked in `set()` forever and the run never started. The run and the
exit each execute on a daemon thread so that a hang fails the test instead of
wedging it; on failure the session's remaining children are killed rather
than joined.
"""

from __future__ import annotations

import threading
import time
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
    f"{name} did not return within {_DEADLINE_S:.0f} s after an idle worker "
    f"had died; it is wedged on a child that no longer exists"
  )
  return outcome.get("error")


def test_second_run_raises_instead_of_hanging(tmp_path: Path) -> None:
  audio = write_marked_audio(tmp_path / "in.wav", 4)
  session = fake_predict_session(tmp_path, n_workers=2)
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  children = [
    *(manager._producer_processes or []),  # noqa: SLF001
    *(manager._worker_processes or []),  # noqa: SLF001
  ]
  left = False
  try:
    session.run([str(audio)])
    time.sleep(0.2)  # the workers are parked in their start-signal wait again
    workers = manager._worker_processes  # noqa: SLF001
    assert workers is not None
    workers[0].kill()
    workers[0].join()

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
