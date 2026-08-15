import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue

from birdnet.acoustic.inference.core.sync import abandon_queue_feeders
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
    start_method: str,
  ) -> None:
    self.__logger: logging.Logger | None = None
    self.__logging_queue = logging_queue
    self.__logging_level = logging_level
    self.__local_queue_handler: QueueHandler | None = None
    self.__name = name
    self.__session_id = session_id
    # The session's effective start method is passed in instead of read from
    # mp.get_start_method(): the global default can differ from the context the
    # pipeline actually uses (e.g. library default "forkserver" while the
    # global still says "fork"), and these branches must match the context the
    # process was really created with.
    self.__start_method = start_method
    self._session_hash = get_session_id_hash(session_id)

  def _init_logging(self) -> None:
    if self.__start_method in ("spawn", "forkserver"):
      init_session_logger(self.__session_id, self.__logging_level)
      self.__local_queue_handler = add_session_queue_handler(
        self.__session_id, self.__logging_queue
      )
    else:
      assert self.__start_method == "fork"
      assert session_queue_handler_exists(self.__session_id, self.__logging_queue)
    self.__logger = get_logger_from_session(self.__session_id, self.__name)
    self.__logger.debug(
      f"Initialized logging for session {self._session_hash} -> {self.__name}."
    )

  def _abandon_logging_feeder(self) -> None:
    """Let this process exit without flushing the logging queue.

    Call last, on the cancellation path only. Every child writes to this one
    queue, and a `multiprocessing.Queue` holds a write lock around the actual
    pipe write on POSIX. A child killed inside that write never releases the
    lock, so every surviving child's logging feeder thread is stuck on it --
    and `multiprocessing` joins that feeder when the process exits, so the
    survivors cannot leave and the parent has to terminate them.

    Only what is still buffered here is dropped, not the records already sent.
    The feeder drains continuously while the parent's log writer is reading, so
    on an ordinary cancellation that tail is next to nothing; in the case this
    exists for, the writer is wedged and could not have read it anyway.
    """
    abandon_queue_feeders(self.__logging_queue)

  def _uninit_logging(self) -> None:
    assert self.__logger is not None
    self.__logger.debug(
      f"Uninitializing logging for session {self._session_hash} -> {self.__name}."
    )
    if self.__start_method in ("spawn", "forkserver"):
      assert self.__local_queue_handler is not None
      remove_session_queue_handler(self.__session_id, self.__local_queue_handler)
    else:
      assert self.__start_method == "fork"
      assert self.__local_queue_handler is None
    self.__local_queue_handler = None
    self.__logger = None

  @property
  def _logger(self) -> logging.Logger:
    assert self.__logger is not None
    return self.__logger
