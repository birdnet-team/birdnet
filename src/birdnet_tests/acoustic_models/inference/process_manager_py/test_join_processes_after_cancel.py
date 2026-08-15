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

from birdnet.acoustic.inference.process_manager import ProcessManager

# Teardown here does no real work: the only waits are the drain-stop timeout and
# the poll interval. Generous enough that a loaded runner cannot flake it, far
# below the "blocked for good" the tests are about.
_TEARDOWN_DEADLINE_S = 30.0


class _FakeProcess:
  """A child that exits on its own after `alive_polls` liveness checks."""

  def __init__(self, name: str, *, alive_polls: int = 0) -> None:
    self.name = name
    self.terminated = False
    self.killed = False
    self._alive_polls = alive_polls

  def is_alive(self) -> bool:
    if self._alive_polls <= 0:
      return False
    self._alive_polls -= 1
    return True

  def join(self, timeout: float | None = None) -> None:
    pass

  def terminate(self) -> None:
    self.terminated = True
    self._alive_polls = 0

  def kill(self) -> None:
    self.killed = True
    self._alive_polls = 0


class _WedgedQueue:
  """A queue whose ``get_nowait`` never returns, like a truncated message."""

  def __init__(self) -> None:
    self.released = threading.Event()
    self.entered = threading.Event()

  def get_nowait(self) -> object:
    self.entered.set()
    self.released.wait()
    raise queue.Empty

  def cancel_join_thread(self) -> None:
    raise AssertionError("a wedged queue must not be touched during close")

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

  _join_processes_after_cancel = ProcessManager._join_processes_after_cancel
  _collect_wedged_drainers = ProcessManager._collect_wedged_drainers
  close_queues = ProcessManager.close_queues


def _logger() -> logging.Logger:
  return logging.getLogger(f"{__name__}.stub")


def test_a_drain_that_never_returns_does_not_stop_teardown() -> None:
  """The whole point: this hangs forever if the drain moves back inline."""
  wedged = _WedgedQueue()
  process = _FakeProcess("Producer-0", alive_polls=2)
  stub = _Stub([process], [wedged])

  try:
    start = time.monotonic()
    stub._join_processes_after_cancel(_logger())
    elapsed_s = time.monotonic() - start
  finally:
    wedged.released.set()

  assert elapsed_s < _TEARDOWN_DEADLINE_S, (
    f"teardown took {elapsed_s:.1f} s with a drain that never returns"
  )
  assert wedged.entered.is_set(), "the queue was never drained, so nothing was tested"
  assert not process.terminated, "the child exited on its own; nothing to terminate"


def test_a_wedged_queue_is_recorded_and_never_closed() -> None:
  """Closing it would free the descriptor number under a blocked read."""
  wedged = _WedgedQueue()
  stub = _Stub([_FakeProcess("Producer-0", alive_polls=1)], [wedged])

  try:
    stub._join_processes_after_cancel(_logger())

    assert any(q is wedged for q in stub._undrainable_queues), (
      "the wedged queue must be remembered so close_queues can skip it"
    )
    # _WedgedQueue raises from close()/cancel_join_thread(), so this only
    # returns if the queue really is skipped.
    stub.close_queues()
  finally:
    wedged.released.set()


def test_healthy_queues_are_drained_and_then_closed() -> None:
  """The drain is what lets a child blocked on a full pipe flush and exit."""
  healthy = _RecordingQueue(["result-a", "result-b"])
  stub = _Stub([_FakeProcess("Producer-0", alive_polls=3)], [healthy])

  stub._join_processes_after_cancel(_logger())

  assert healthy.drained == ["result-a", "result-b"]
  assert not stub._undrainable_queues

  stub.close_queues()
  assert healthy.closed


def test_a_child_that_never_exits_is_terminated_after_the_grace_period() -> None:
  stubborn = _FakeProcess("Worker-0", alive_polls=10_000)
  healthy = _RecordingQueue()
  stub = _Stub([stubborn], [healthy])

  stub._join_processes_after_cancel(_logger(), grace_period_s=0.2)

  assert stubborn.terminated, "a child outliving the grace period must be terminated"
