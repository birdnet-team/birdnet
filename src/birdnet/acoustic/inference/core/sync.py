from __future__ import annotations

import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from types import TracebackType


class CountedSemaphore:
  """
  Drop-in replacement for ``mp.Semaphore`` whose ``get_value()`` works on
  macOS by mirroring acquire/release into a shared counter.
  """

  def __init__(self, initial: int = 0) -> None:
    self._sem = mp.Semaphore(initial)
    self._counter: Synchronized = mp.Value("i", initial)

  def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
    acquired = self._sem.acquire(block, timeout)
    if acquired:
      with self._counter.get_lock():
        self._counter.value -= 1
    return acquired

  def release(self) -> None:
    with self._counter.get_lock():
      self._counter.value += 1
    self._sem.release()

  def get_value(self) -> int:
    return self._counter.value

  def __enter__(self) -> bool:
    return self.acquire()

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: TracebackType | None,
  ) -> None:
    self.release()
