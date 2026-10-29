"""A cancelled child flushes its buffered log records before abandoning the feeder.

A child that logs a burst faster than the parent's writer reads it and then
exits finds the pipe full and its feeder thread holding the tail: the records
the parent has not read yet. Abandoning the feeder at that moment drops the
tail wholesale, and a feeder cut off inside a write leaves a half-written
frame that wedges the parent's reader for good. The reader here starts late to
stand in for a writer that is behind.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from multiprocessing import Queue

import pytest

import birdnet.acoustic.inference.core.logs as logs_module
from birdnet.acoustic.inference.core.logs import LogableProcessBase
from birdnet.acoustic.inference.core.sync import ThreadedQueueReader

pytestmark = pytest.mark.no_tf

# Well past any pipe: 64 KB on Linux, 8 KB on Windows, at ~300 bytes per
# pickled record.
N_RECORDS = 3_000
# Long enough for the burst to fill the pipe before anyone reads it.
READER_DELAY_S = 0.5
DEADLINE_S = 30.0


def _log_burst_then_abandon(log_queue: Queue, session_id: str) -> None:
  child = LogableProcessBase(session_id, "burst", log_queue, logging.DEBUG, "spawn")
  child._init_logging()
  for i in range(N_RECORDS):
    child._logger.debug(f"record {i} " + "x" * 200)
  child._uninit_logging()
  child._abandon_logging_feeder()


def _count_records(
  reader: ThreadedQueueReader, child: mp.process.BaseProcess
) -> tuple[int, float]:
  """Records received, and how long after the last one the child was seen dead."""
  n = 0
  last_at = time.monotonic()
  deadline = time.monotonic() + DEADLINE_S
  while time.monotonic() < deadline:
    try:
      item = reader.get(timeout=0.2)
    except queue.Empty:
      if not child.is_alive():
        dead_at = time.monotonic()
        # One more pass for what the pump still had in flight.
        try:
          while True:
            reader.get_nowait()
            n += 1
            last_at = dead_at
        except queue.Empty:
          pass
        return n, dead_at - last_at
      continue
    if isinstance(item, logging.LogRecord):
      n += 1
      last_at = time.monotonic()
  return n, DEADLINE_S


def _burst_run() -> tuple[int, float, int | None]:
  """Records received, seconds from the last one to the child's death, exit code."""
  ctx = mp.get_context("spawn")
  log_queue = ctx.Queue()
  child = ctx.Process(
    target=_log_burst_then_abandon, args=(log_queue, "drain-test"), daemon=True
  )
  child.start()
  time.sleep(READER_DELAY_S)
  reader: ThreadedQueueReader = ThreadedQueueReader(log_queue, "drain-test-reader")
  counted: dict[str, tuple[int, float]] = {}
  t = threading.Thread(
    target=lambda: counted.__setitem__("n", _count_records(reader, child)),
    daemon=True,
  )
  t.start()
  t.join(timeout=DEADLINE_S + 5)
  try:
    assert not t.is_alive(), "the reader never finished: a torn frame wedged it"
    child.join(timeout=5)
    n, exit_lag = counted.get("n", (0, DEADLINE_S))
    return n, exit_lag, child.exitcode
  finally:
    if child.is_alive():
      child.kill()
    reader.stop()
    # A child that exited inside a write leaves the queue's write lock held;
    # without this the interpreter would join this side's feeder at exit and
    # block on that lock, turning a failed test into a hung session.
    log_queue.cancel_join_thread()


def test_records_buffered_at_exit_reach_the_parent() -> None:
  n, elapsed, exitcode = _burst_run()
  assert exitcode == 0
  # "initialized"/"uninitialized" lines come on top of the burst.
  assert n >= N_RECORDS, (
    f"{n} of {N_RECORDS} records arrived: the child abandoned its logging "
    f"feeder with the tail still buffered"
  )


def test_the_child_leaves_as_soon_as_its_records_are_read() -> None:
  """The drain ends when the feeder is empty, not at its 2 s cap.

  Measured from the last record's arrival to the child's death, so a slow
  runner's spawn and import time does not count; fastest of two against jitter.
  A prompt child measures one reader poll (0.2 s), a drain that always waits
  its cap about 1.8 s.
  """
  fastest = min(_burst_run()[1] for _ in range(2))
  assert fastest < logs_module._LOG_DRAIN_TIMEOUT_S - 1.0, (
    f"the child was alive {fastest:.2f} s after its last record"
  )


def test_a_child_nobody_reads_still_leaves_within_the_drain_bound() -> None:
  ctx = mp.get_context("spawn")
  log_queue = ctx.Queue()
  child = ctx.Process(
    target=_log_burst_then_abandon, args=(log_queue, "drain-unread"), daemon=True
  )
  start = time.monotonic()
  child.start()
  child.join(timeout=DEADLINE_S)
  elapsed = time.monotonic() - start
  try:
    assert not child.is_alive(), "the child never gave up on its logging feeder"
    assert child.exitcode == 0
    assert elapsed >= logs_module._LOG_DRAIN_TIMEOUT_S - 0.1
  finally:
    if child.is_alive():
      child.kill()
    log_queue.cancel_join_thread()
