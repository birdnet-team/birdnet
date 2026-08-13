"""A child that dies without signalling must be reported, not waited on.

Every parent-side wait in a healthy run is unbounded on purpose. That is only
safe while the children are working: one killed by the OOM killer or crashing
in native code never sets its finish signal, so the parent would block forever
with no output. These tests pin the decision logic that turns that into an
error; the end-to-end proof is in
`acoustic_models/v2_4/model_py/test_predict/test_dead_worker_v2_4.py`.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager


class _FakeProcess:
  def __init__(self, name: str, *, alive: bool, exitcode: int | None = None) -> None:
    self.name = name
    self.exitcode = exitcode
    self._alive = alive

  def is_alive(self) -> bool:
    return self._alive


class _StubManager:
  """Only what `raise_if_child_died` touches: pairs, logger, cancel event."""

  def __init__(self, pairs: list[tuple[_FakeProcess, threading.Event]]) -> None:
    self._pairs = pairs
    self._logger = logging.getLogger(f"{__name__}.stub")
    self.cancel_event = threading.Event()
    self._child_death_error: ChildProcessError | None = None
    self._res = SimpleNamespace(
      processing_resources=SimpleNamespace(cancel_event=self.cancel_event)
    )

  def _iter_children_with_finish_signals(self):  # noqa: ANN202
    return iter(self._pairs)

  raise_if_child_died = ProcessManager.raise_if_child_died
  _wait_for_finish_signal = ProcessManager._wait_for_finish_signal


class _PairsStub:
  """Exercises the real `_iter_children_with_finish_signals`.

  The manager attributes it reads are set directly, so the pairing logic and
  its `strict=True` invariant are tested rather than replaced by a stub.
  """

  def __init__(
    self,
    producers: list[_FakeProcess] | None,
    producer_signals: list[threading.Event],
    workers: list[_FakeProcess] | None,
    worker_signals: list[threading.Event],
    tracker: _FakeProcess | None = None,
    tracker_signal: threading.Event | None = None,
  ) -> None:
    self._producer_processes = producers
    self._worker_processes = workers
    self._perf_tracker_process = tracker
    self._res = SimpleNamespace(
      producer_resources=SimpleNamespace(finish_signals=producer_signals),
      worker_resources=SimpleNamespace(finish_signals=worker_signals),
      stats_resources=SimpleNamespace(perf_res_finish_signal=tracker_signal),
    )

  _iter_children_with_finish_signals = ProcessManager._iter_children_with_finish_signals


def _signal(*, is_set: bool) -> threading.Event:
  event = threading.Event()
  if is_set:
    event.set()
  return event


def test_running_children_are_not_reported() -> None:
  manager = _StubManager(
    [
      (_FakeProcess("Worker-0", alive=True), _signal(is_set=False)),
      (_FakeProcess("Producer-0", alive=True), _signal(is_set=False)),
    ]
  )

  manager.raise_if_child_died()


def test_child_that_exited_after_signalling_is_not_reported() -> None:
  """The normal end-of-session shutdown: signalled, then exited."""
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=False, exitcode=0), _signal(is_set=True))]
  )

  manager.raise_if_child_died()


def test_child_killed_without_signalling_is_reported() -> None:
  manager = _StubManager(
    [
      (_FakeProcess("Worker-0", alive=True), _signal(is_set=False)),
      # -9 is what the OOM killer leaves behind.
      (_FakeProcess("Worker-1", alive=False, exitcode=-9), _signal(is_set=False)),
    ]
  )

  with pytest.raises(ChildProcessError, match=r"Worker-1.*-9"):
    manager.raise_if_child_died()


def test_child_that_exited_zero_without_signalling_is_reported() -> None:
  """Exit code 0 is still wrong mid-run: nobody is left to finish the work."""
  manager = _StubManager(
    [(_FakeProcess("Producer-0", alive=False, exitcode=0), _signal(is_set=False))]
  )

  with pytest.raises(ChildProcessError, match="Producer-0"):
    manager.raise_if_child_died()


def test_no_children_is_not_an_error() -> None:
  _StubManager([]).raise_if_child_died()


def test_reporting_a_dead_child_marks_the_run_cancelled() -> None:
  """The cancel event routes teardown through the queue-draining join.

  `wait_until_all_finished` has no exception handler of its own, so without
  this the error would escape with the run still marked healthy and `join()`
  would take the path that waits for buffered data a dead child never sends --
  moving the hang to teardown instead of removing it.
  """
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=False, exitcode=-9), _signal(is_set=False))]
  )
  assert not manager.cancel_event.is_set()

  with pytest.raises(ChildProcessError):
    manager.raise_if_child_died()

  assert manager.cancel_event.is_set()


def test_healthy_children_do_not_mark_the_run_cancelled() -> None:
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=True), _signal(is_set=False))]
  )

  manager.raise_if_child_died()

  assert not manager.cancel_event.is_set()


def test_a_child_that_shut_down_cleanly_is_not_blamed_on_memory() -> None:
  """Exit code 0 means the child saw cancel/end while parked and left.

  Reached when a session whose run was cancelled is used again: every child is
  already gone and `reset()` has cleared the finish signals. Reporting that as
  an OOM kill would send users tuning `n_workers` for no reason.
  """
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=False, exitcode=0), _signal(is_set=False))]
  )

  with pytest.raises(ChildProcessError, match="can no longer run") as excinfo:
    manager.raise_if_child_died()

  assert "OOM" not in str(excinfo.value)
  assert "n_workers" not in str(excinfo.value)


def test_the_reported_error_is_stored_for_the_session_to_surface() -> None:
  """The consumer swallows the exception, so the session reads it from here."""
  manager = _StubManager(
    [(_FakeProcess("Worker-1", alive=False, exitcode=-9), _signal(is_set=False))]
  )

  with pytest.raises(ChildProcessError) as excinfo:
    manager.raise_if_child_died()

  assert manager._child_death_error is excinfo.value


# --- the real pairing logic (no stub for _iter_children_with_finish_signals) --


def test_pairing_covers_producers_workers_and_the_tracker() -> None:
  stub = _PairsStub(
    [_FakeProcess("P0", alive=True)],
    [_signal(is_set=False)],
    [_FakeProcess("W0", alive=True), _FakeProcess("W1", alive=True)],
    [_signal(is_set=False), _signal(is_set=False)],
    _FakeProcess("PT", alive=True),
    _signal(is_set=False),
  )

  names = [p.name for p, _ in stub._iter_children_with_finish_signals()]

  assert names == ["P0", "W0", "W1", "PT"]


def test_pairing_skips_process_groups_that_were_never_started() -> None:
  stub = _PairsStub(None, [], None, [], None, None)

  assert list(stub._iter_children_with_finish_signals()) == []


def test_pairing_fails_loudly_when_counts_diverge() -> None:
  """strict=True: zipping to the shorter list would silently drop a child."""
  stub = _PairsStub(
    [_FakeProcess("P0", alive=True), _FakeProcess("P1", alive=True)],
    [_signal(is_set=False)],  # one signal for two producers
    None,
    [],
  )

  with pytest.raises(ValueError, match="argument"):
    list(stub._iter_children_with_finish_signals())


# --- _wait_for_finish_signal: it must give up, never raise and never block ---
#
# Both exits below were added because the caller (`session._run`) always calls
# `_raise_if_cancelled` straight after, which turns the stored error into the
# same RuntimeError every other failure path produces. Mutation testing showed
# neither exit was covered: deleting either left the whole suite green.


def _never_set() -> threading.Event:
  return threading.Event()


def test_wait_gives_up_when_a_child_dies_rather_than_raising() -> None:
  """Raising here would surface an OSError where callers expect RuntimeError.

  It must also not keep waiting: the signal in this test is never set, so a
  wait that ignored the dead child would hang the test.
  """
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=False, exitcode=-9), _signal(is_set=False))]
  )

  manager._wait_for_finish_signal(_never_set())

  assert manager.cancel_event.is_set(), "the run must be marked cancelled"
  assert manager._child_death_error is not None, (
    "the diagnosis must be stored for the session to surface"
  )


def test_wait_gives_up_when_the_run_was_cancelled() -> None:
  """ProgressDispatcher sets the cancel event and returns without signalling.

  It is a thread, so the liveness check cannot see it; without this exit the
  wait for its finish signal would never return.
  """
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=True), _signal(is_set=False))]
  )
  manager.cancel_event.set()

  manager._wait_for_finish_signal(_never_set())

  assert manager._child_death_error is None, (
    "a plain cancellation must not be reported as a dead child"
  )


def test_wait_returns_promptly_once_the_signal_is_set() -> None:
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=True), _signal(is_set=False))]
  )

  manager._wait_for_finish_signal(_signal(is_set=True))

  assert not manager.cancel_event.is_set()
