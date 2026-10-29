"""The log writer only ever reads the logging queue.

A parent-side put goes through the queue's write lock, which a child killed
inside a log write leaves held for good on POSIX: a writer that put its own
end-of-drain marker into the queue then never got it back and blocked for
the rest of the teardown. The writer must reach the records without putting
anything, and must notice the stop event on its own.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import uuid
from pathlib import Path

import pytest

from birdnet.acoustic.inference.file_writer import QueueFileWriter

pytestmark = pytest.mark.no_tf

# A record is read within the writer's idle poll; a stop within the next one.
RECORD_DEADLINE_S = 5.0
STOP_DEADLINE_S = 5.0


def test_records_reach_the_file_without_a_parent_side_put(tmp_path: Path) -> None:
  ctx = mp.get_context("spawn")
  log_queue = ctx.Queue()
  original_put = log_queue.put
  puts: list[object] = []

  def recording_put(obj: object, *args: object, **kwargs: object) -> None:
    puts.append(obj)
    original_put(obj, *args, **kwargs)  # type: ignore[arg-type]

  log_queue.put = recording_put  # type: ignore[method-assign]
  stop_event = threading.Event()
  log_file = tmp_path / "session.log"
  writer = QueueFileWriter(
    session_id=uuid.uuid4().hex,
    log_queue=log_queue,
    logging_level=logging.DEBUG,
    log_file=log_file,
    cancel_event=ctx.Event(),
    stop_event=stop_event,
  )
  thread = threading.Thread(target=writer, name="writer-under-test", daemon=True)
  thread.start()
  try:
    record = logging.LogRecord(
      "test", logging.INFO, __file__, 1, "hello %s", ("world",), None
    )
    original_put(record)
    deadline = time.monotonic() + RECORD_DEADLINE_S
    while time.monotonic() < deadline:
      if log_file.exists() and "hello world" in log_file.read_text(encoding="utf-8"):
        break
      time.sleep(0.02)
    else:
      pytest.fail("the record never reached the session log")

    stop_event.set()
    thread.join(timeout=STOP_DEADLINE_S)
    assert not thread.is_alive(), "the writer did not stop after the stop event"
    assert puts == [], f"the writer put into the logging queue: {puts!r}"
  finally:
    stop_event.set()
    log_queue.close()
    log_queue.join_thread()
