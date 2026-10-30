"""The wait for the progress dispatcher ends however the dispatcher does.

The dispatcher reports its own failures and cancels the run, so the wait's own
watch on the thread is a fallback that a real run does not reach. These cases
pin it, and the deadline for the tracker's closing update, on a stub that
borrows the real methods.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference import process_manager
from birdnet.acoustic.inference.process_manager import (
  PipelineThreadError,
  ProcessManager,
)

pytestmark = pytest.mark.no_tf

# Patched down from 30 s and 1 s, so a case takes a fraction of a second; the
# deadline spans many polls.
_DEADLINE_S = 0.15
_POLL_S = 0.01
# How long the dispatcher stays inside the callback in the busy case: several
# deadlines.
_BUSY_FOR_S = 4 * _DEADLINE_S
# How long the dispatcher lives before it ends in the phase-2 case.
_DIES_AFTER_S = 0.1
# The test's own bound on one wait, so a wait that never returns fails the test.
_BOUND_S = 10.0


class _FakeThread:
  def __init__(self, *, alive: bool) -> None:
    self.name = "abcde-ProgressDispatcher"
    self.alive = alive

  def is_alive(self) -> bool:
    return self.alive


class _FakeTracker:
  name = "abcde-PerformanceTracker"
  exitcode = None

  def is_alive(self) -> bool:
    return True


class _StubManager:
  """Only what the dispatcher wait touches."""

  def __init__(self, thread: _FakeThread) -> None:
    self._progress_dispatcher_thread = thread
    self._perf_tracker_process = _FakeTracker()
    self._logger = logging.getLogger(f"{__name__}.stub")
    self.cancel_event = threading.Event()
    self._child_death_error: ChildProcessError | None = None
    self._run_error: BaseException | None = None
    self._run_error_lock = threading.Lock()
    self._res = SimpleNamespace(
      processing_resources=SimpleNamespace(cancel_event=self.cancel_event)
    )
    self.last_update = threading.Event()
    self.busy = threading.Event()
    self.finish = threading.Event()

  _wait_for_progress_dispatcher = ProcessManager._wait_for_progress_dispatcher
  _poll_progress_dispatcher = ProcessManager._poll_progress_dispatcher
  _record_dispatcher_death = ProcessManager._record_dispatcher_death
  _record_lost_last_update = ProcessManager._record_lost_last_update
  record_run_error = ProcessManager.record_run_error
  record_child_death = ProcessManager.record_child_death


@pytest.fixture(autouse=True)
def _short_waits(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", _DEADLINE_S)
  monkeypatch.setattr(process_manager, "_LIVENESS_POLL_INTERVAL_S", _POLL_S)


def _wait(manager: _StubManager) -> tuple[float, float]:
  """Run the wait on a thread; return when it started and when it returned."""
  done = threading.Event()
  errors: list[BaseException] = []

  def call() -> None:
    try:
      manager._wait_for_progress_dispatcher(
        manager.last_update, manager.busy, manager.finish
      )
    except BaseException as e:  # noqa: BLE001
      errors.append(e)
    finally:
      done.set()

  started = time.monotonic()
  threading.Thread(target=call, name="wait", daemon=True).start()
  returned = done.wait(_BOUND_S)
  if not returned:
    # Ends the stuck wait with the test.
    manager.cancel_event.set()
  assert returned, f"the wait did not return within {_BOUND_S:.0f} s"
  assert not errors, errors
  return started, time.monotonic()


def _later(delay_s: float, action: Callable[[], None]) -> None:
  timer = threading.Timer(delay_s, action)
  timer.daemon = True
  timer.start()


def test_a_dead_dispatcher_fails_the_run_before_the_deadline() -> None:
  thread = _FakeThread(alive=False)
  manager = _StubManager(thread)

  _wait(manager)

  error = manager._run_error
  assert isinstance(error, PipelineThreadError), error
  assert str(error) == (
    f"Progress dispatcher thread '{thread.name}' ended without delivering the "
    f"performance tracker's last progress update and reported no error."
  )
  assert manager.cancel_event.is_set()
  # Set only at the deadline.
  assert manager._child_death_error is None


def test_time_inside_the_callback_does_not_count_against_the_deadline() -> None:
  manager = _StubManager(_FakeThread(alive=True))
  manager.busy.set()
  left_the_callback_at: list[float] = []

  def leave_the_callback_and_finish() -> None:
    left_the_callback_at.append(time.monotonic())
    manager.busy.clear()
    manager.last_update.set()
    manager.finish.set()

  _later(_BUSY_FOR_S, leave_the_callback_and_finish)

  _, returned = _wait(manager)

  # Same-clock ordering rather than a floor on the elapsed time: the timer is
  # armed before the wait starts, and its own wait can end milliseconds short.
  assert left_the_callback_at and returned >= left_the_callback_at[0]
  assert manager._child_death_error is None, manager._child_death_error
  assert manager._run_error is None, manager._run_error
  assert not manager.cancel_event.is_set()


def test_a_closing_update_that_never_arrives_fails_the_run_at_the_deadline() -> None:
  manager = _StubManager(_FakeThread(alive=True))

  started, returned = _wait(manager)

  assert returned - started >= _DEADLINE_S
  error = manager._child_death_error
  assert isinstance(error, ChildProcessError), error
  assert f"Pipeline process '{_FakeTracker.name}' finished its work" in str(error)
  assert "still alive" in str(error)
  assert manager._run_error is None, manager._run_error
  assert manager.cancel_event.is_set()


def test_a_dispatcher_that_ends_inside_the_closing_callback_fails_the_run() -> None:
  """Phase 2 has no deadline, so only the watch on the thread can end it."""
  thread = _FakeThread(alive=True)
  manager = _StubManager(thread)
  manager.last_update.set()

  def die() -> None:
    thread.alive = False

  _later(_DIES_AFTER_S, die)

  _wait(manager)

  error = manager._run_error
  assert isinstance(error, PipelineThreadError), error
  assert str(error) == (
    f"Progress dispatcher thread '{thread.name}' ended without finishing after "
    f"the performance tracker's last progress update and reported no error."
  )
  assert manager.cancel_event.is_set()
  assert manager._child_death_error is None
