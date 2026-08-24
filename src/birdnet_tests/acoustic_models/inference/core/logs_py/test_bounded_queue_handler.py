"""Oversized log records must be cut before crossing the process boundary.

A record above ~16 KB is pickled into a frame POSIX writes as separate header
and body, which is the widest window for a killed writer to leave the session
log's reader a half-written message -- and the records that big are exception
dumps with ``stack_info``, exactly what a dying child emits. The cut removes
the *middle*: the tail of a formatted exception record carries the exception
type and message, the one line that names what went wrong.
"""

from __future__ import annotations

import io
import logging
import queue
from multiprocessing.reduction import ForkingPickler

import pytest

from birdnet.acoustic.inference.core.logs import (
  _MAX_RECORD_CHARS,
  BoundedQueueHandler,
)

pytestmark = pytest.mark.no_tf

# POSIX splits a queue message into separate header and body writes above this.
_SINGLE_WRITE_BYTES = 16_384


def _send_through_handler(record: logging.LogRecord) -> logging.LogRecord:
  buffer: queue.Queue = queue.Queue()
  handler = BoundedQueueHandler(buffer)
  handler.emit(record)
  return buffer.get_nowait()


def _record(msg: str) -> logging.LogRecord:
  return logging.LogRecord(
    name="birdnet.test",
    level=logging.ERROR,
    pathname=__file__,
    lineno=1,
    msg=msg,
    args=None,
    exc_info=None,
  )


def test_a_short_record_crosses_untouched() -> None:
  received = _send_through_handler(_record("a perfectly ordinary line"))
  assert received.msg == "a perfectly ordinary line"


def test_a_giant_record_keeps_its_head_and_its_last_line() -> None:
  """The tail is where the diagnosis lives; dropping it keeps only the noise."""
  head = "worker exception follows"
  filler = "\n".join(f"  File 'frame_{i}.py', line {i}" for i in range(4000))
  tail = "MemoryError: could not allocate the batch"
  received = _send_through_handler(_record(f"{head}\n{filler}\n{tail}"))

  assert received.msg.startswith(head), "the log message itself must survive"
  assert received.msg.endswith(tail), (
    "the exception line at the end is the diagnosis and must survive the cut"
  )
  assert "[log record truncated;" in received.msg
  assert len(received.msg) <= _MAX_RECORD_CHARS + 200


def test_a_giant_record_pickles_to_a_single_write() -> None:
  """Ties the cap to the constant it exists for."""
  received = _send_through_handler(_record("x" * 200_000))
  buf = io.BytesIO()
  ForkingPickler(buf).dump(received)
  assert len(buf.getvalue()) <= _SINGLE_WRITE_BYTES, (
    f"a truncated record still pickles to {len(buf.getvalue())} bytes, past "
    f"the single-write threshold the cap exists to stay under"
  )


def test_a_rendered_exception_dump_is_what_gets_cut() -> None:
  """End to end through logger machinery, with a real exception and stack."""
  buffer: queue.Queue = queue.Queue()
  handler = BoundedQueueHandler(buffer)
  logger = logging.getLogger("birdnet.test.bounded")
  logger.setLevel(logging.DEBUG)
  logger.addHandler(handler)
  logger.propagate = False
  try:
    try:
      raise ValueError("the actual reason" + "!" * 20_000)
    except ValueError:
      logger.exception("worker died", stack_info=True)
  finally:
    logger.removeHandler(handler)

  received = buffer.get_nowait()
  assert len(received.msg) <= _MAX_RECORD_CHARS + 200
  assert received.msg.startswith("worker died")
