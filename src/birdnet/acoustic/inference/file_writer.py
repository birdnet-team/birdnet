import logging
import multiprocessing.synchronize
import queue
import threading
from logging.handlers import MemoryHandler
from multiprocessing import Queue
from pathlib import Path


class QueueFileWriter:
  def __init__(
    self,
    session_id: str,
    log_queue: Queue,
    logging_level: int,
    log_file: Path,
    cancel_event: multiprocessing.synchronize.Event,
    stop_event: threading.Event,
  ) -> None:
    self._session_id = session_id
    self._logging_level = logging_level
    self._log_queue = log_queue
    self._log_file = log_file
    self._cancel_event = cancel_event
    self._logging_stop_event = stop_event
    # Only an idle writer waits this long: a record wakes the read at once. It
    # bounds how long the stop event goes unnoticed, so it is part of every exit.
    self._get_logs_interval_s = 0.1
    self._logger = logging.getLogger(f"birdnet_file_writer.session_{self._session_id}")
    self._logger.setLevel(self._logging_level)
    self._logger.propagate = False

  def __call__(self) -> None:
    f = logging.Formatter(
      "%(asctime)s %(processName)-10s %(message)s",
      "%H:%M:%S",
    )

    h = logging.FileHandler(self._log_file, mode="w", encoding="utf-8")

    LARGE_LOG_SIZE_THAT_WILL_NOT_BE_REACHED = 100_000
    mh = MemoryHandler(
      capacity=LARGE_LOG_SIZE_THAT_WILL_NOT_BE_REACHED,
      flushLevel=logging.WARNING,
      target=h,
      flushOnClose=True,
    )

    h.setFormatter(f)
    assert len(self._logger.handlers) == 0
    self._logger.addHandler(mh)

    try:
      self._run_main_loop()
    finally:
      self._logger.removeHandler(mh)
      mh.close()
      # MemoryHandler.close() drops its target without closing it.
      h.close()

  def _run_main_loop(self) -> None:
    assert len(self._logger.handlers) == 1
    memory_handler = self._logger.handlers[0]
    # Reads with a timeout and never puts anything into the queue itself: a
    # parent-side put goes through the queue's write lock, which a child killed
    # inside a write leaves held for good (POSIX). Reading continuously also
    # keeps the pipe empty, so a child's feeder holds that lock for the
    # microseconds of a write rather than for the length of a parent-side pause.
    while True:
      try:
        try:
          record: logging.LogRecord = self._log_queue.get(
            timeout=self._get_logs_interval_s
          )
        except queue.Empty:
          memory_handler.flush()
          if self._logging_stop_event.is_set():
            self._logger.debug("Stop logging event set and log queue is empty.")
            break
          continue
        self._logger.handle(record)
      except OSError as e:
        # Expected when the file is closed mid-write, e.g. on termination.
        if e.args[0] == "handle is closed":
          self._cancel_event.set()
          break
      except EOFError:
        print("EOFError: Queue was closed, stopping file writer.")
        self._cancel_event.set()
        break
      except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping file writer.")
        self._cancel_event.set()
        break
      except Exception:
        import sys
        import traceback

        print("Problem:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        self._cancel_event.set()
        break
