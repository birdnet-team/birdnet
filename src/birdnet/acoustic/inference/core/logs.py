import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue

from birdnet.acoustic.inference.core.sync import FailureSlot, write_failure
from birdnet.core.base import get_session_id_hash
from birdnet.utils.logging_utils import get_package_logger, init_package_logger


def get_session_logger(session_id: str) -> logging.Logger:
  logger_name = f"birdnet.session_{session_id}"
  logger = logging.getLogger(logger_name)
  logger.parent = get_package_logger()
  return logger


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
  assert handler in root.handlers
  root.removeHandler(handler)


# Rendered size above which a record is cut before crossing the process
# boundary -- a byte budget, not characters: the threshold it stays under is
# in bytes, and a CJK or emoji character is up to four of them. 12 000 bytes
# of UTF-8 plus the record's other attributes pickles safely inside one
# `send_bytes` call (the split into separate header and body writes starts at
# 16 KB), which narrows -- not closes; a full pipe can still cut any write
# mid-frame -- the window for a killed writer to leave the log reader a
# half-written record. Records this size are exception dumps with
# `stack_info`: exactly what a dying child emits, and the reader of the
# session log is bounded either way.
_MAX_RECORD_BYTES = 12_000
# The cut removes the middle, not the tail: a formatted exception record ends
# with the exception type and message -- the one line that names what went
# wrong -- while the head carries the log message and the innermost frames.
_TRUNCATED_HEAD_BYTES = 7_000
_TRUNCATED_TAIL_BYTES = 4_000
# How long a child waits for its logging feeder to empty before it exits
# without it. The parent's writer reads continuously, so this covers the drain
# itself plus a writer briefly behind; measured need 0.2 to 0.4 s.
_LOG_DRAIN_TIMEOUT_S = 2.0


class BoundedQueueHandler(QueueHandler):
  """A QueueHandler whose records have a bounded rendered size."""

  def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
    record = super().prepare(record)
    # After super().prepare the whole payload -- message, arguments, rendered
    # traceback -- lives in record.msg as one string.
    if isinstance(record.msg, str):
      # surrogatepass on encode: paths from the OS can carry lone surrogates,
      # and a log handler must never raise. `ignore` on decode drops at most
      # a clipped character at each cut point.
      encoded = record.msg.encode("utf-8", "surrogatepass")
      if len(encoded) > _MAX_RECORD_BYTES:
        dropped = len(encoded) - _TRUNCATED_HEAD_BYTES - _TRUNCATED_TAIL_BYTES
        record.msg = (
          encoded[:_TRUNCATED_HEAD_BYTES].decode("utf-8", "ignore")
          + f"\n... [log record truncated; {dropped} bytes dropped] ...\n"
          + encoded[-_TRUNCATED_TAIL_BYTES:].decode("utf-8", "ignore")
        )
        record.message = record.msg
    return record


def add_session_queue_handler(session_id: str, logging_queue: Queue) -> QueueHandler:
  root = get_session_logger(session_id)
  h = BoundedQueueHandler(logging_queue)
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
    failure_slot: FailureSlot | None = None,
  ) -> None:
    self.__logger: logging.Logger | None = None
    self.__logging_queue = logging_queue
    self.__failure_slot = failure_slot
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

  def _report_failure(self, exc: BaseException) -> None:
    """Leave the exception's type and message where the parent can read them."""
    if self.__failure_slot is not None:
      write_failure(self.__failure_slot, exc)

  def _abandon_logging_feeder(self) -> None:
    """Flush the logging feeder for a bounded time, then exit without it.

    Call last, on every exit path. Every child writes to this one queue, and a
    `multiprocessing.Queue` holds a write lock around the actual pipe write on
    POSIX. A child killed inside that write never releases the lock, so every
    surviving child's logging feeder thread is stuck on it -- and
    `multiprocessing` joins that feeder when the process exits, so without
    this bound the survivors cannot leave, whether their run ended cleanly or
    not, and the parent has to terminate them.

    What the feeder still holds is whatever the parent's writer has not read
    yet: on a cancel that includes the child's own error record. And a feeder
    cut off inside a write leaves a half-written frame that blocks the
    parent's writer for good, and leaves the queue's write lock held, which
    blocks every other process's next log write. So the feeder gets a bounded
    chance to finish; only when nothing reads the pipe is the tail abandoned.
    """
    q = self.__logging_queue
    # Hands the feeder its sentinel; nothing is logged after _uninit_logging.
    q.close()
    feeder = getattr(q, "_thread", None)
    if feeder is not None:
      feeder.join(timeout=_LOG_DRAIN_TIMEOUT_S)
    q.cancel_join_thread()

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
