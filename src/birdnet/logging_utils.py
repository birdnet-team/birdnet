# birdnet/logging_utils.py
from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.synchronize
import threading
import time
from logging.handlers import QueueHandler
from multiprocessing import Queue
from pathlib import Path
from time import sleep

from birdnet.globals import PKG_NAME
from birdnet.io_lock import IOLockHandler, LockedMemoryHandler


def get_package_logger():
  return logging.getLogger(PKG_NAME)


# # The worker configuration is done at the start of the worker process run.
# # Note that on Windows you can't rely on fork semantics, so each process
# # will run the logging configuration code when it starts.
# def process_logging_configurer(logging_queue: Queue):
#   root = logging.getLogger()
#   assert root.level == logging.WARNING
#   assert root.hasHandlers() is False
#   h = QueueHandler(logging_queue)  # Just the one handler needed
#   root.setLevel(logging.NOTSET)
#   root.addHandler(h)


def add_queue_handler(logging_queue: Queue):
  root = get_package_logger()
  h = QueueHandler(logging_queue)  # Just the one handler needed
  root.addHandler(h)
  return h


def queue_handler_exists(logging_queue: Queue):
  root = get_package_logger()
  for handler in root.handlers:
    if isinstance(handler, QueueHandler) and handler.queue is logging_queue:
      return True
  return False


def remove_queue_handler(handler: QueueHandler):
  root = get_package_logger()
  # check has queue handler already
  assert handler in root.handlers
  root.removeHandler(handler)


def get_logger(name: str):
  logger = logging.getLogger(name)
  logger.parent = get_package_logger()
  return logger


def get_package_logging_level() -> int:
  """
  Gibt das Logging-Level des birdnet-Pakets zurück.
  """
  result = get_package_logger().level
  return result


def init_package_logger(logging_level: int) -> None:
  root = get_package_logger()
  root.setLevel(logging_level)
  root.propagate = False


init_package_logger(logging.INFO)


class QueueFileWriter:
  def __init__(
    self,
    log_queue: Queue,
    logging_level: int,
    log_file: Path,
    io_lock_handler: IOLockHandler,
    cancel_event: multiprocessing.synchronize.Event,
    stop_event: threading.Event,
  ):
    self._logging_level = logging_level
    self._log_queue = log_queue
    self._log_file = log_file
    self._io_log_handler = io_lock_handler
    self._cancel_event = cancel_event
    self._stop_event = stop_event
    self._get_logs_interval = 5

  def __call__(self):
    logger = logging.getLogger("birdnet-file-writer")
    logger.setLevel(self._logging_level)
    logger.propagate = False
    assert len(logger.handlers) == 0

    # log to temp
    f = logging.Formatter(
      "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )

    h = logging.FileHandler(self._log_file, mode="w", encoding="utf-8")
    mh = LockedMemoryHandler(
      capacity=10000,
      io_lock_handler=self._io_log_handler,
      flush_interval_s=15,
      flushLevel=logging.WARNING,
      target=h,
      flushOnClose=True,
    )

    h.setFormatter(f)
    logger.addHandler(mh)

    while not self._stop_event.is_set():
      try:
        perf_c = time.perf_counter()
        while not self._log_queue.empty():
          record: logging.LogRecord = self._log_queue.get()
          logger.handle(record)
        get_que_duration = time.perf_counter() - perf_c
        logger.debug(f"{get_que_duration}s to get logging entries from queue.")
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
      sleep(self._get_logs_interval)
    mh.flush()


class LogableProcessBase:
  def __init__(
    self,
    name: str,
    logging_queue: mp.Queue,
    logging_level: int,
  ):
    self.__logger: logging.Logger | None = None
    self.__logging_queue = logging_queue
    self.__logging_level = logging_level
    self.__local_queue_handler: QueueHandler | None = None
    self.__name = name

  def _init_logging(self) -> None:
    if mp.get_start_method() in ("spawn", "forkserver"):
      init_package_logger(self.__logging_level)
      self.__local_queue_handler = add_queue_handler(self.__logging_queue)
    else:
      assert mp.get_start_method() == "fork"
      assert queue_handler_exists(self.__logging_queue)
    self.__logger = get_logger(self.__name)
    self.__logger.debug(f"Initialized logging for {self.__name}.")

  def _uninit_logging(self) -> None:
    assert self.__logger is not None
    self.__logger.debug(f"Uninitializing logging for {self.__name}.")
    if mp.get_start_method() in ("spawn", "forkserver"):
      assert self.__local_queue_handler is not None
      remove_queue_handler(self.__local_queue_handler)
    else:
      assert mp.get_start_method() == "fork"
      assert self.__local_queue_handler is None
    self.__local_queue_handler = None
    self.__logger = None

  @property
  def _logger(self) -> logging.Logger:
    assert self.__logger is not None
    return self.__logger
