"""A queue reader must bound its caller even when the queue cannot be read.

`Queue.get`'s timeout covers the read lock and the poll but not the receive,
so a message a killed child left half-written blocks the calling thread for
good, with nothing raised (issues #77/#83). `ThreadedQueueReader` confines
that block to a sacrificial daemon thread; these tests pin that the *caller*
stays bounded in every case, including against a queue genuinely poisoned by
a killed writer.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import time
from queue import Empty

import pytest

from birdnet.acoustic.inference.core.sync import ThreadedQueueReader
from birdnet_tests.torn_queue import make_torn_queue

pytestmark = pytest.mark.no_tf

# Bounds "the caller returned", not how fast a machine is.
_CALLER_BOUND_S = 10.0


def test_items_arrive_in_order_with_real_timeouts() -> None:
  ctx = mp.get_context("spawn")
  q = ctx.Queue()
  for i in range(5):
    q.put(i)

  reader = ThreadedQueueReader(q, "order-reader")
  try:
    got = [reader.get(timeout=5.0) for _ in range(5)]
    assert got == [0, 1, 2, 3, 4]
    start = time.monotonic()
    with pytest.raises(Empty):
      reader.get(timeout=0.2)
    assert time.monotonic() - start < _CALLER_BOUND_S
  finally:
    assert reader.close(), "a healthy reader must stop when asked"
  q.cancel_join_thread()
  q.close()


def test_a_wedged_pump_is_reported_and_never_blocks_the_caller() -> None:
  """A fake queue that blocks forever stands in for the torn frame.

  Cross-platform and cheap; the real poisoned queue is the test below.
  """
  release = threading.Event()

  class _BlockingQueue:
    def get(self, timeout: float | None = None) -> object:
      release.wait(60.0)
      raise Empty

  reader = ThreadedQueueReader(_BlockingQueue(), "wedged-reader")  # type: ignore[arg-type]
  try:
    start = time.monotonic()
    with pytest.raises(Empty):
      reader.get(timeout=0.5)
    assert time.monotonic() - start < _CALLER_BOUND_S, (
      "the caller waited on the queue instead of the buffer"
    )
    assert reader.close(timeout=0.5) is False, (
      "a pump stuck inside the queue must be reported as wedged"
    )
  finally:
    release.set()


def test_an_undeserializable_payload_fails_the_caller_instead_of_the_pump() -> None:
  """The pump must not die silently, leaving the caller on an empty buffer."""

  class _CorruptQueue:
    def get(self, timeout: float | None = None) -> object:
      raise EOFError("stand-in for any deserialization failure")

  reader = ThreadedQueueReader(_CorruptQueue(), "corrupt-reader")  # type: ignore[arg-type]
  with pytest.raises(RuntimeError, match="could not be deserialized"):
    reader.get(timeout=_CALLER_BOUND_S)
  assert reader.close()


def test_a_queue_poisoned_by_a_killed_writer_cannot_block_the_caller() -> None:
  """The real thing: a writer killed mid-frame, not a stand-in.

  On POSIX the pipe then holds a length header whose body never arrives, and a
  plain ``Queue.get`` would block on it forever -- the pump must wedge there
  *instead of the caller*. On Windows message-mode pipes cannot tear, so the
  same kill loses messages cleanly; the caller-side assertions are identical,
  which is the point of the abstraction.
  """
  q, _writer = make_torn_queue()

  reader = ThreadedQueueReader(q, "poisoned-reader")
  start = time.monotonic()
  drained = 0
  outcome = "empty"
  try:
    while True:
      try:
        reader.get(timeout=2.0)
        drained += 1
      except Empty:
        break
      except RuntimeError:
        # A cleanly lost tail can surface as a broken-pipe deserialization
        # failure on some platforms; for the caller that is equally bounded.
        outcome = "error"
        break
  finally:
    elapsed = time.monotonic() - start
    pump_stopped = reader.close()

  assert elapsed < 3 * _CALLER_BOUND_S, (
    f"the caller was blocked for {elapsed:.0f} s by a killed writer's queue"
  )
  if sys.platform != "win32":
    # The frame is torn here, so the pump must be stuck inside _recv_bytes on
    # the partial body -- proving this test poisoned the queue for real rather
    # than passing on a clean pipe.
    assert pump_stopped is False, (
      f"the pump was expected to wedge on the torn frame (drained={drained}, "
      f"outcome={outcome}); a clean stop means the scenario was not created"
    )
  # The queue is deliberately not closed when the pump is wedged: closing it
  # would free its descriptor for reuse underneath the blocked read.
  if pump_stopped:
    q.cancel_join_thread()
    q.close()
