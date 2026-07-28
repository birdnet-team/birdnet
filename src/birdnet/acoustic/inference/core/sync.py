from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.context import BaseContext
from multiprocessing.sharedctypes import Synchronized
from types import TracebackType


def abandon_queue_feeders(*queues: Queue | None) -> None:
  """Let the current process exit without flushing these queues' feeders.

  On the cancellation path the parent stops reading the child->parent queues, so
  any items still buffered in a child's feeder thread would otherwise block the
  child's shutdown: its exit handler joins the feeder, which is stuck writing to
  a full pipe that nobody drains. ``cancel_join_thread`` drops that buffered data
  and lets the process exit immediately. Losing the data is fine here -- the run
  was cancelled and the results are discarded anyway.

  Exiting cleanly this way (instead of being force-terminated by the parent) also
  avoids leaking the semaphores the child inherited: a killed process never runs
  the finalizers that unregister them from the multiprocessing resource tracker.

  Only call this once cancellation is certain; ``None`` entries are skipped so
  callers can pass optional queues directly.
  """
  for q in queues:
    if q is not None:
      q.cancel_join_thread()


class CountedSemaphore:
  """
  Drop-in replacement for ``mp.Semaphore`` whose ``get_value()`` works on
  macOS by mirroring acquire/release into a shared counter.
  """

  def __init__(self, initial: int = 0, ctx: BaseContext | None = None) -> None:
    # The context must match the one the pipeline's processes are created
    # with; primitives from mismatched contexts cannot be shared reliably.
    ctx = ctx if ctx is not None else mp.get_context()
    self._sem = ctx.Semaphore(initial)
    self._counter: Synchronized = ctx.Value("i", initial)

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
