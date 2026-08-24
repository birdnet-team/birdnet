"""Messages promised by finished children must be waited for with a deadline.

The senders set their finish signals before these reads happen, so the
liveness check can no longer flag one that died on its way out -- and ``put``
hands off to a feeder thread, so a child killed *after* signalling can die
with its message unsent or half-written. Before this, the parent read these
queues with ``get(block=True, timeout=None)``: a wait nothing could end.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference import process_manager
from birdnet.acoustic.inference.process_manager import ProcessManager

pytestmark = pytest.mark.no_tf

# The patched deadline is 1 s; this bounds "returned at the deadline", not speed.
_CALLER_BOUND_S = 20.0


class _Stub:
  def __init__(self) -> None:
    self._session_hash = "testhash"
    self._logger = logging.getLogger(f"{__name__}.stub")
    self._undrainable_queues: list[object] = []
    self._child_death_error: ChildProcessError | None = None
    self.cancel_event = threading.Event()
    self._res = SimpleNamespace(
      processing_resources=SimpleNamespace(cancel_event=self.cancel_event),
      logging_resources=SimpleNamespace(session_log_file=Path("stub-session.log")),
    )

  read_promised = ProcessManager.read_promised
  _close_reader = ProcessManager._close_reader


@pytest.fixture
def _short_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
  # 30 s is right for a session and wrong for a test. Not autouse: the happy
  # path needs feeder threads and a pump to spin up, and running it under a
  # one-second budget would make a slow runner read as a broken fix.
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", 1.0)


def test_messages_that_arrived_are_returned_in_order() -> None:
  ctx = mp.get_context("spawn")
  q = ctx.Queue()
  q.put({1, 2})
  q.put({3})

  stub = _Stub()
  out = stub.read_promised(q, 2, "test reports")

  assert out == [{1, 2}, {3}]
  assert not stub.cancel_event.is_set()
  q.cancel_join_thread()
  q.close()


@pytest.mark.usefixtures("_short_deadline")
def test_a_missing_message_fails_at_the_deadline_instead_of_waiting_forever() -> None:
  """The regression: one producer died after signalling, before delivering."""
  ctx = mp.get_context("spawn")
  q = ctx.Queue()
  q.put({1})  # one of two promised reports

  stub = _Stub()
  start = time.monotonic()
  with pytest.raises(RuntimeError, match="Only 1 of 2 test reports"):
    stub.read_promised(q, 2, "test reports")
  elapsed = time.monotonic() - start

  assert elapsed < _CALLER_BOUND_S, (
    f"took {elapsed:.0f} s: the deadline is not being applied"
  )
  # The run must be marked failed, or the caller would carry on into result
  # assembly with a producer's report silently missing.
  assert stub.cancel_event.is_set()
  # ...and diagnosed like every other child death: stored for the session to
  # surface, and pointing the user at the log.
  assert isinstance(stub._child_death_error, ChildProcessError)
  assert "check the logs" in str(stub._child_death_error)
  q.cancel_join_thread()
  q.close()


def test_a_cancelled_run_stops_the_wait_immediately() -> None:
  ctx = mp.get_context("spawn")
  q = ctx.Queue()

  stub = _Stub()
  stub.cancel_event.set()
  with pytest.raises(RuntimeError, match="cancelled"):
    stub.read_promised(q, 1, "test reports")
  q.cancel_join_thread()
  q.close()


def test_a_wedged_reader_keeps_its_queue_out_of_the_closable_set() -> None:
  """`close_queues` must skip a queue whose pump is still inside a read."""
  import queue as pyqueue

  from birdnet.acoustic.inference.core.sync import ThreadedQueueReader

  release = threading.Event()

  class _BlockingQueue:
    def get(self, timeout: float | None = None) -> object:
      release.wait(60.0)
      raise pyqueue.Empty

  stub = _Stub()
  q = _BlockingQueue()
  reader = ThreadedQueueReader(q, "wedge-registration")  # type: ignore[arg-type]
  try:
    stub._close_reader(reader, q)  # type: ignore[arg-type]
    assert any(q is seen for seen in stub._undrainable_queues), (
      "a wedged reader's queue must be recorded so close_queues leaves it open"
    )
    # Closing the same reader's queue twice must not record it twice.
    stub._close_reader(reader, q)  # type: ignore[arg-type]
    assert sum(1 for seen in stub._undrainable_queues if q is seen) == 1
  finally:
    release.set()
