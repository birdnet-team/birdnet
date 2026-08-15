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

import pytest

from birdnet.acoustic.inference import process_manager
from birdnet.acoustic.inference.process_manager import ProcessManager

# How long the stuck thread stays stuck. Comfortably past the (patched) join
# timeout, so an unbounded join is released late enough to fail the assertions
# rather than hang the suite until pytest-timeout kills the xdist worker.
_STUCK_S = 5.0
_PATCHED_JOIN_TIMEOUT_S = 0.2
# The join should return after the patched timeout; this only has to separate
# "returned on the timeout" from "waited out the stuck thread".
_RETURN_DEADLINE_S = _STUCK_S / 2


class _Queue:
  """Stands in for the queue the stuck thread is blocked reading."""


class _Stub:
  def __init__(self) -> None:
    self._undrainable_queues: list[object] = []

  _join_reader_thread = ProcessManager._join_reader_thread


def _logger() -> logging.Logger:
  return logging.getLogger(f"{__name__}.stub")


@pytest.fixture(autouse=True)
def _short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
  # The real bound is 30 s, which is right for a session and wrong for a test.
  monkeypatch.setattr(
    process_manager, "_READER_JOIN_TIMEOUT_S", _PATCHED_JOIN_TIMEOUT_S
  )


def test_a_thread_that_finishes_is_joined_and_nothing_is_recorded() -> None:
  stub = _Stub()
  q = _Queue()
  thread = threading.Thread(target=lambda: None, name="quick-reader", daemon=True)
  thread.start()

  stub._join_reader_thread(_logger(), thread, q)

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
    stub._join_reader_thread(_logger(), thread, q)
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
    stub._join_reader_thread(_logger(), thread, None)
  finally:
    release.set()

  assert not stub._undrainable_queues
