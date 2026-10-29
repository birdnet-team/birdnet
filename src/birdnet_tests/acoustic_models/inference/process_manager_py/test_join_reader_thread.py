"""Teardown must not wait forever on a thread reading a child-to-parent queue.

`ProgressDispatcher` reads its queue with `get(block=True, timeout=..)`, and
that timeout bounds acquiring the read lock and the poll -- not `_recv_bytes`.
So a performance tracker killed mid-`put` leaves a message that thread can
never finish reading, with nothing raised (issue #77). Joining it without a
bound moves the hang from the drain to the join.
"""

from __future__ import annotations

import logging
import threading
import time

from birdnet.acoustic.inference.core.sync import Deadline
from birdnet.acoustic.inference.process_manager import ProcessManager

# How long the stuck thread stays stuck. Comfortably past the deadline handed
# to the join, so an unbounded join is released late enough to fail the
# assertions rather than hang the suite until pytest-timeout kills the worker.
_STUCK_S = 5.0
# The deadline handed to the join; the real one is the session's teardown budget.
_JOIN_TIMEOUT_S = 0.2
# The join should return on its deadline; this only has to separate "returned
# on the deadline" from "waited out the stuck thread".
_RETURN_DEADLINE_S = _STUCK_S / 2


class _Queue:
  """Stands in for the queue the stuck thread is blocked reading."""


class _Stub:
  def __init__(self) -> None:
    self._undrainable_queues: list[object] = []

  _join_reader_thread = ProcessManager._join_reader_thread


def _logger() -> logging.Logger:
  return logging.getLogger(f"{__name__}.stub")


def test_a_thread_that_finishes_is_joined_and_nothing_is_recorded() -> None:
  stub = _Stub()
  q = _Queue()
  thread = threading.Thread(target=lambda: None, name="quick-reader", daemon=True)
  thread.start()

  stub._join_reader_thread(_logger(), thread, q, Deadline(_JOIN_TIMEOUT_S))

  assert not stub._undrainable_queues, "a healthy reader's queue must stay closeable"


def test_a_thread_that_never_returns_does_not_stop_teardown() -> None:
  stub = _Stub()
  q = _Queue()
  release = threading.Event()
  thread = threading.Thread(
    target=lambda: release.wait(_STUCK_S), name="stuck-reader", daemon=True
  )
  thread.start()

  try:
    start = time.monotonic()
    stub._join_reader_thread(_logger(), thread, q, Deadline(_JOIN_TIMEOUT_S))
    elapsed_s = time.monotonic() - start
  finally:
    release.set()

  assert elapsed_s < _RETURN_DEADLINE_S, (
    f"the join took {elapsed_s:.1f} s: it waited for the reader instead of bounding it"
  )
  assert any(recorded is q for recorded in stub._undrainable_queues), (
    "the queue the stuck reader holds must be left open, not closed under it"
  )


def test_a_stuck_thread_without_a_queue_records_nothing() -> None:
  """The completion dispatcher reads an in-process queue, which cannot truncate."""
  stub = _Stub()
  release = threading.Event()
  thread = threading.Thread(
    target=lambda: release.wait(_STUCK_S), name="stuck-reader", daemon=True
  )
  thread.start()

  try:
    stub._join_reader_thread(_logger(), thread, None, Deadline(_JOIN_TIMEOUT_S))
  finally:
    release.set()

  assert not stub._undrainable_queues
