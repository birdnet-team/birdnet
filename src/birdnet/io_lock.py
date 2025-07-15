import logging
import multiprocessing.synchronize
import threading
from contextlib import nullcontext
from logging.handlers import MemoryHandler


class IOLockHandler:
  def __init__(self, serial_io: bool, io_lock: multiprocessing.synchronize.Lock | None):
    self._serial_io = serial_io
    self._io_lock = io_lock

  def __enter__(self):
    if self._serial_io:
      assert self._io_lock is not None
      return self._io_lock.__enter__()
    return nullcontext()

  def __exit__(self, exc_type, exc_value, traceback):
    if self._serial_io:
      assert self._io_lock is not None
      return self._io_lock.__exit__(exc_type, exc_value, traceback)
    return False


class LockedMemoryHandler(MemoryHandler):
  """
  MemoryHandler, der beim Flush einen externen mp.Lock benutzt.
  """

  def __init__(
    self,
    capacity,
    io_lock_handler: IOLockHandler,
    flushLevel=logging.ERROR,
    target=None,
    flushOnClose=True,
  ):
    super().__init__(capacity, flushLevel, target, flushOnClose)
    self._io_lock_handler = io_lock_handler

  def flush(self):
    with self._io_lock_handler:
      super().flush()


class LockedMemoryHandlerWithTimer(MemoryHandler):
  """
  MemoryHandler, der beim Flush einen externen mp.Lock benutzt.
  """

  def __init__(
    self,
    capacity,
    io_lock_handler: IOLockHandler,
    flush_interval_s=30,
    flushLevel=logging.ERROR,
    target=None,
    flushOnClose=True,
  ):
    super().__init__(capacity, flushLevel, target, flushOnClose)
    # derselbe Lock für alle Prozesse!
    self._io_lock_handler = io_lock_handler

    self._stop_evt = threading.Event()
    # >>> Start des Hintergrund-Threads
    self.flush_interval_s = flush_interval_s
    th = threading.Thread(
      target=self._continous_flush,
      name="LockedMemoryHandler.ContinousFlush",
      daemon=True,
    )
    th.start()
    self._thread = th

  def flush(self):
    with self._io_lock_handler:
      super().flush()

  def _continous_flush(self):
    while not self._stop_evt.wait(self.flush_interval_s):
      self.flush()

  def close(self):
    self._stop_evt.set()
    self._thread.join()
    super().close()
