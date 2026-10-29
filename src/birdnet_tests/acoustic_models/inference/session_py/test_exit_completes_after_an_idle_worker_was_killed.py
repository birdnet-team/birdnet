"""Leaving a session completes even if an idle worker was just killed.

`end()` releases every idle child's start signal. A start signal is a
semaphore because a release never blocks; `multiprocessing.Event.set()`
waits for every registered waiter to confirm it woke, forever when that waiter
was killed inside `Event.wait()`, and `is_alive()` still reports a just-killed
child as alive, so no liveness check closes that window. The exit runs on a
daemon thread so that a hang fails the test instead of wedging it; on failure
the session's remaining children are killed rather than joined.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_DEADLINE_S = 20.0


def test_exit_returns_after_killing_an_idle_worker(tmp_path: Path) -> None:
  audio = write_marked_audio(tmp_path / "in.wav", 4)
  session = fake_predict_session(tmp_path, n_workers=2)
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  children = [
    *(manager._producer_processes or []),  # noqa: SLF001
    *(manager._worker_processes or []),  # noqa: SLF001
  ]
  done = threading.Event()
  errors: list[BaseException] = []
  try:
    session.run([str(audio)])
    time.sleep(0.2)  # the workers are parked in their start-signal wait again
    workers = manager._worker_processes  # noqa: SLF001
    assert workers is not None
    workers[0].kill()
    # Deliberately not joined: the parent must not depend on seeing the death.

    def leave() -> None:
      try:
        session.__exit__(None, None, None)
      except BaseException as e:  # noqa: BLE001
        errors.append(e)
      finally:
        done.set()

    threading.Thread(target=leave, name="session-exit", daemon=True).start()

    assert done.wait(_DEADLINE_S), (
      f"__exit__ did not return within {_DEADLINE_S:.0f} s after an idle worker "
      f"was killed; the teardown is wedged on a child that no longer exists"
    )
    assert errors == []
  finally:
    if not done.is_set():
      for child in children:
        child.kill()
