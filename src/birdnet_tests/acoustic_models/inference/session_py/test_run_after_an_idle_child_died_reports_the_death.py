"""A session reused after an idle child process died fails that run with a diagnosis.

Starting a run releases every child's start signal. A semaphore release
returns even when the child was killed inside its wait, so the run proceeds
and the liveness check reports the dead process; a `multiprocessing.Event`
would block `set()` until the dead waiter confirmed it woke. Producers,
workers and the performance tracker each wait on a start signal of their own,
so each kind is killed in turn. The run and the exit each execute on a daemon
thread so that a hang fails the test instead of wedging it; on failure the
session's remaining children are killed rather than joined.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
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


def _within_deadline(
  step: Callable[[], object], name: str, kind: str
) -> BaseException | None:
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
    f"{name} did not return within {_DEADLINE_S:.0f} s after an idle {kind} "
    f"had died; it is wedged on a child that no longer exists"
  )
  return outcome.get("error")


@pytest.mark.parametrize("kind", KINDS.keys())
def test_second_run_raises_instead_of_hanging(tmp_path: Path, kind: str) -> None:
  audio = write_marked_audio(tmp_path / "in.wav", 4)
  session = fake_predict_session(tmp_path, n_workers=2, **KINDS[kind])  # type: ignore[arg-type]
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  children = _child_processes(manager)
  left = False
  try:
    session.run([str(audio)])
    time.sleep(0.2)  # the children are parked in their start-signal wait again
    child = _idle_child(manager, kind)
    child.kill()
    child.join()

    error = _within_deadline(lambda: session.run([str(audio)]), "run()", kind)
    assert isinstance(error, RuntimeError), error
    assert "Pipeline process" in str(error)

    assert (
      _within_deadline(lambda: session.__exit__(None, None, None), "__exit__", kind)
      is None
    )
    left = True
  finally:
    if not left:
      for remaining in children:
        remaining.kill()
