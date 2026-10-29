"""Teardown after a cancelled run must be bounded, whatever the queues do.

The cancel path drains the child-to-parent queues so children blocked on a full
pipe can flush and exit. ``Queue.get_nowait`` is not actually non-blocking: it
checks that *some* bytes are available and then waits for the rest of the
frame, so a child killed mid-``put`` leaves a truncated message that stops the
call for good -- with no exception to catch, because the surviving children
still hold the pipe's write end (issue #77).

These tests pin that the drain cannot stop teardown, and that a queue whose
drain is wedged is left open instead of being closed underneath the blocked
read.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from types import SimpleNamespace

from birdnet.acoustic.inference import process_manager
from birdnet.acoustic.inference.core.sync import Deadline
from birdnet.acoustic.inference.process_manager import ProcessManager

# Teardown here does no real work: the only waits are the drain-stop timeout
# (1 s) and the joins of fake processes. Kept close to that so a wait that
# outlives the budget is caught, but loose enough for a loaded runner.
_TEARDOWN_DEADLINE_S = 10.0
# Budget handed to the method under test; the joins return at once.
_BUDGET_S = 8.0


class _FakeProcess:
  """A child that exits when joined, unless told to stay alive."""

  def __init__(
    self, name: str, *, exits: bool = True, ignores_terminate: bool = False
  ) -> None:
    self.name = name
    self.terminated = False
    self.killed = False
    self._exits = exits
    self._ignores_terminate = ignores_terminate
    self._alive = True

  def is_alive(self) -> bool:
    return self._alive

  def join(self, timeout: float | None = None) -> None:
    if self._exits:
      self._alive = False
    else:
      # Like a real join on a child that stays: it waits the timeout out.
      time.sleep(min(timeout if timeout is not None else 1.0, 1.0))

  def terminate(self) -> None:
    self.terminated = True
    if not self._ignores_terminate:
      self._alive = False

  def kill(self) -> None:
    self.killed = True
    self._alive = False


class _WedgedQueue:
  """A queue whose ``get_nowait`` never returns, like a truncated message."""

  def __init__(self) -> None:
    self.released = threading.Event()
    self.entered = threading.Event()
    self.join_cancelled = False

  def get_nowait(self) -> object:
    self.entered.set()
    self.released.wait()
    raise queue.Empty

  def cancel_join_thread(self) -> None:
    self.join_cancelled = True

  def close(self) -> None:
    raise AssertionError("a wedged queue must not be closed")


class _RecordingQueue:
  """A healthy queue that hands out `items` and is then empty."""

  def __init__(self, items: list[object] | None = None) -> None:
    self.items = list(items or [])
    self.drained: list[object] = []
    self.closed = False

  def get_nowait(self) -> object:
    if not self.items:
      raise queue.Empty
    item = self.items.pop(0)
    self.drained.append(item)
    return item

  def cancel_join_thread(self) -> None:
    pass

  def close(self) -> None:
    self.closed = True


class _Stub:
  """Only what the two methods under test touch."""

  def __init__(
    self,
    processes: list[_FakeProcess],
    queues: list[object],
  ) -> None:
    self._session_hash = "testhash"
    self._producer_processes = list(processes)
    self._worker_processes: list[_FakeProcess] = []
    self._perf_tracker_process = None
    self._undrainable_queues: list[object] = []
    self._stopped_readers: list = []
    # The manager reads the queues off the resources by name and filters out the
    # ones a run did not create, so the slots this test does not need are None.
    padded = [*queues, *[None] * 7]
    self._res = SimpleNamespace(
      producer_resources=SimpleNamespace(
        input_queue=None, unprocessed_inputs_queue=padded[0]
      ),
      worker_resources=SimpleNamespace(results_queue=padded[1]),
      file_completion_resources=SimpleNamespace(marker_queue=padded[2]),
      logging_resources=SimpleNamespace(logging_queue=None),
      stats_resources=SimpleNamespace(
        wkr_stats_queue=padded[3],
        prd_stats_queue=padded[4],
        perf_res_queue=padded[5],
        callback_queue=padded[6],
      ),
    )

    self._teardown = None

  _join_processes_after_cancel = ProcessManager._join_processes_after_cancel
  _collect_wedged_drainers = ProcessManager._collect_wedged_drainers
  _reap_children = ProcessManager._reap_children
  _children = ProcessManager._children
  close_queues = ProcessManager.close_queues
  reap_stopped_readers = ProcessManager.reap_stopped_readers


def _logger() -> logging.Logger:
  return logging.getLogger(f"{__name__}.stub")


def test_a_drain_that_never_returns_does_not_stop_teardown() -> None:
  """The whole point: teardown must not wait on a drain that never returns.

  The release timer is what makes this a *failing* test rather than a hanging
  one if the drain moves back onto the teardown path. Without it the assertion
  below could never be evaluated -- the call would simply never return, and
  pytest-timeout would take the whole xdist worker with it.
  """
  wedged = _WedgedQueue()
  process = _FakeProcess("Producer-0")
  stub = _Stub([process], [wedged])

  # Comfortably past the deadline, so a teardown that does wait for the drain
  # is released late enough to fail the assertion rather than pass slowly.
  release = threading.Timer(_TEARDOWN_DEADLINE_S * 2, wedged.released.set)
  release.daemon = True
  release.start()
  try:
    start = time.monotonic()
    stub._join_processes_after_cancel(_logger(), Deadline(_BUDGET_S))
    elapsed_s = time.monotonic() - start
  finally:
    release.cancel()
    wedged.released.set()

  assert elapsed_s < _TEARDOWN_DEADLINE_S, (
    f"teardown took {elapsed_s:.1f} s with a drain that never returns"
  )
  assert wedged.entered.is_set(), "the queue was never drained, so nothing was tested"
  assert not process.terminated, "the child exited on its own; nothing to terminate"


def test_a_wedged_queue_is_recorded_and_never_closed() -> None:
  """Closing it would free the descriptor number under a blocked read."""
  wedged = _WedgedQueue()
  stub = _Stub([_FakeProcess("Producer-0")], [wedged])

  try:
    stub._join_processes_after_cancel(_logger(), Deadline(_BUDGET_S))

    assert any(q is wedged for q in stub._undrainable_queues), (
      "the wedged queue must be remembered so close_queues can skip it"
    )
    # _WedgedQueue raises from close(), so this only returns if it is skipped.
    stub.close_queues()
    # ...but the writer-side join must still be cancelled, or the interpreter
    # joins this queue's feeder thread at exit and blocks there.
    assert wedged.join_cancelled, (
      "cancel_join_thread must be called even on a queue that is left open"
    )
  finally:
    wedged.released.set()


def test_healthy_queues_are_drained_and_then_closed() -> None:
  """The drain is what lets a child blocked on a full pipe flush and exit."""
  healthy = _RecordingQueue(["result-a", "result-b"])
  stub = _Stub([_FakeProcess("Producer-0")], [healthy])

  stub._join_processes_after_cancel(_logger(), Deadline(_BUDGET_S))

  assert healthy.drained == ["result-a", "result-b"]
  assert not stub._undrainable_queues

  stub.close_queues()
  assert healthy.closed


def test_a_child_that_never_exits_is_terminated_after_the_grace_period() -> None:
  stubborn = _FakeProcess("Worker-0", exits=False)
  healthy = _RecordingQueue()
  stub = _Stub([stubborn], [healthy])

  # The grace is the budget less the kill reserve: 0.2 s here.
  start = time.monotonic()
  stub._join_processes_after_cancel(
    _logger(), Deadline(process_manager._KILL_RESERVE_S + 0.2)
  )
  elapsed_s = time.monotonic() - start

  assert stubborn.terminated, "a child outliving the grace period must be terminated"
  assert not stubborn.killed
  assert 0.2 <= elapsed_s < _TEARDOWN_DEADLINE_S, (
    f"grace not honoured: {elapsed_s:.2f} s"
  )


def test_a_child_that_ignores_termination_is_killed() -> None:
  deaf = _FakeProcess("Worker-0", exits=False, ignores_terminate=True)
  stub = _Stub([deaf], [_RecordingQueue()])

  stub._join_processes_after_cancel(_logger(), Deadline(0.5))

  assert deaf.terminated
  assert deaf.killed, "a child that ignores SIGTERM must be killed, not waited for"
  assert not deaf.is_alive()
