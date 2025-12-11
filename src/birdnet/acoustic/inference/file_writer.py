import logging
import multiprocessing.synchronize
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
    processing_finished_event: multiprocessing.synchronize.Event,
  ) -> None:
    self._session_id = session_id
    self._logging_level = logging_level
    self._log_queue = log_queue
    self._log_file = log_file
    self._cancel_event = cancel_event
    self._logging_stop_event = stop_event
    self._get_logs_interval_s = 1
    self._processing_finished_event = processing_finished_event
    self._logger = logging.getLogger(f"birdnet_file_writer.session_{self._session_id}")
    self._logger.setLevel(self._logging_level)
    self._logger.propagate = False

  def __call__(self) -> None:
    f = logging.Formatter(
      "%(asctime)s %(processName)-10s %(message)s",
      # "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
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

    self._run_main_loop()

    # print(time.time(), "flushing on close")
    mh.close()

    # print(time.time(), "done flushing")
    # lines = self._log_file.read_text(encoding="utf-8").splitlines()
    # sorted_lines = sorted(lines, key=lambda x: x.split()[:2])
    # self._log_file.write_text("\n".join(sorted_lines), encoding="utf-8")
    # print(
    #   f"Finished writing logs to {self._log_file.absolute()}.
    # Total lines: {len(sorted_lines)}."
    # )

  def _run_main_loop(self) -> None:
    assert len(self._logger.handlers) == 1
    memory_handler = self._logger.handlers[0]
    stop_logging = False
    while True:
      try:
        get_end_marker = None
        self._log_queue.put(get_end_marker)

        # NOTE: !empty() not working reliable and qsize() not available on macOS
        is_empty = True
        n_received = 0

        while True:
          queue_entry = self._log_queue.get(block=True)
          if is_end_marker := queue_entry is get_end_marker:
            break
          is_empty = False
          n_received += 1

          record: logging.LogRecord = queue_entry
          self._logger.handle(record)
        # print("Received", n_received, "log records.")
        memory_handler.flush()

        if is_empty:
          if not stop_logging:
            stop_logging = self._logging_stop_event.wait(self._get_logs_interval_s)
          else:
            self._logger.debug("Stop logging event set and log queue is empty.")
            break
      except OSError as e:
        # OSError can happen if the file is closed while writing
        if e.args[0] == "handle is closed":
          # This is expected if the file is closed while writing
          # e.g., when the process is terminated
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
      # if not self._logging_stop_event.is_set():
      # sleep(self._get_logs_interval)
