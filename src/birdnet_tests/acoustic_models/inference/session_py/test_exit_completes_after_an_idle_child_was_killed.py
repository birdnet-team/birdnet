"""Leaving a session completes even if an idle child process was just killed.

`end()` releases every idle child's start signal. A start signal is a
semaphore because a release never blocks; `multiprocessing.Event.set()`
waits for every registered waiter to confirm it woke, forever when that waiter
was killed inside `Event.wait()`, and `is_alive()` still reports a just-killed
child as alive, so no liveness check closes that window. Producers, workers
and the performance tracker each wait on a start signal of their own, so each
kind is killed in turn. The exit runs on a daemon thread so that a hang fails
the test instead of wedging it; on failure the session's remaining children
are killed rather than joined.
"""

from __future__ import annotations

import threading
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_DEADLINE_S = 20.0

# Session options per kind of child; the performance tracker process only
# exists with `show_stats`.
KINDS = {
  "producer": {},
  "worker": {},
  "tracker": {"show_stats": "progress"},
}


def _child_processes(manager: ProcessManager) -> list[BaseProcess]:
  tracker = manager._perf_tracker_process  # noqa: SLF001
  return [
    *(manager._producer_processes or []),  # noqa: SLF001
    *(manager._worker_processes or []),  # noqa: SLF001
    *([tracker] if tracker is not None else []),
  ]


def _idle_child(manager: ProcessManager, kind: str) -> BaseProcess:
  if kind == "producer":
    processes = manager._producer_processes  # noqa: SLF001
  elif kind == "worker":
    processes = manager._worker_processes  # noqa: SLF001
  else:
    tracker = manager._perf_tracker_process  # noqa: SLF001
    processes = [tracker] if tracker is not None else []
  assert processes, kind
  return processes[0]


@pytest.mark.parametrize("kind", KINDS.keys())
def test_exit_returns_after_killing_an_idle_child(tmp_path: Path, kind: str) -> None:
  audio = write_marked_audio(tmp_path / "in.wav", 4)
  session = fake_predict_session(tmp_path, n_workers=2, **KINDS[kind])  # type: ignore[arg-type]
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  children = _child_processes(manager)
  done = threading.Event()
  errors: list[BaseException] = []
  try:
    session.run([str(audio)])
    time.sleep(0.2)  # the children are parked in their start-signal wait again
    _idle_child(manager, kind).kill()
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
      f"__exit__ did not return within {_DEADLINE_S:.0f} s after an idle {kind} "
      f"was killed; the teardown is wedged on a child that no longer exists"
    )
    assert errors == []
  finally:
    if not done.is_set():
      for child in children:
        child.kill()
