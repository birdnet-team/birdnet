import logging
import multiprocessing as mp
import multiprocessing.synchronize
from logging.handlers import QueueHandler
from multiprocessing import Queue

from birdnet.core.base import get_session_id_hash
from birdnet.utils.logging_utils import get_package_logger, init_package_logger


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
