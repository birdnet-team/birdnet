import logging
import multiprocessing as mp
import multiprocessing.synchronize
import threading
from logging.handlers import MemoryHandler, QueueHandler
from multiprocessing import Queue
from pathlib import Path

from birdnet.base import get_session_id_hash
from birdnet.logging_utils import get_package_logger, init_package_logger


def get_session_logger(session_id: str) -> logging.Logger:
  logger_name = f"birdnet.session_{session_id}"
  logger = logging.getLogger(logger_name)
  logger.parent = get_package_logger()
  return logger


def get_session_logging_level(session_id: str) -> int:
  result = get_session_logger(session_id).level
  return result


def init_session_logger(session_id: str, logging_level: int) -> None:
  init_package_logger(logging_level)
  root = get_session_logger(session_id)
  root.setLevel(logging_level)
  root.propagate = False


def get_logger_from_session(session_id: str, name: str) -> logging.Logger:
  session_logger = get_session_logger(session_id)
  logger = logging.getLogger(f"{session_logger.name}.{name}")
  logger.parent = session_logger
  return logger


def remove_session_queue_handler(session_id: str, handler: QueueHandler) -> None:
  root = get_session_logger(session_id)
  # check has queue handler already
  assert handler in root.handlers
  root.removeHandler(handler)


def add_session_queue_handler(session_id: str, logging_queue: Queue) -> QueueHandler:
  root = get_session_logger(session_id)
  h = QueueHandler(logging_queue)  # Just the one handler needed
  root.addHandler(h)
  return h


def session_queue_handler_exists(session_id: str, logging_queue: Queue) -> bool:
  root = get_session_logger(session_id)
  for handler in root.handlers:
    if isinstance(handler, QueueHandler) and handler.queue is logging_queue:
      return True
  return False


class LogableProcessBase:
  def __init__(
    self,
    session_id: str,
    name: str,
    logging_queue: Queue,
    logging_level: int,
  ) -> None:
    self.__logger: logging.Logger | None = None
    self.__logging_queue = logging_queue
    self.__logging_level = logging_level
    self.__local_queue_handler: QueueHandler | None = None
    self.__name = name
    self.__session_id = session_id
    self._session_hash = get_session_id_hash(session_id)

  def _init_logging(self) -> None:
    if mp.get_start_method() in ("spawn", "forkserver"):
      init_session_logger(self.__session_id, self.__logging_level)
      self.__local_queue_handler = add_session_queue_handler(
        self.__session_id, self.__logging_queue
      )
    else:
      assert mp.get_start_method() == "fork"
      assert session_queue_handler_exists(self.__session_id, self.__logging_queue)
    self.__logger = get_logger_from_session(self.__session_id, self.__name)
    self.__logger.debug(
      f"Initialized logging for session {self._session_hash} -> {self.__name}."
    )

  def _uninit_logging(self) -> None:
    assert self.__logger is not None
    self.__logger.debug(
      f"Uninitializing logging for session {self._session_hash} -> {self.__name}."
    )
    if mp.get_start_method() in ("spawn", "forkserver"):
      assert self.__local_queue_handler is not None
      remove_session_queue_handler(self.__session_id, self.__local_queue_handler)
    else:
      assert mp.get_start_method() == "fork"
      assert self.__local_queue_handler is None
    self.__local_queue_handler = None
    self.__logger = None

  @property
  def _logger(self) -> logging.Logger:
    assert self.__logger is not None
    return self.__logger


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
