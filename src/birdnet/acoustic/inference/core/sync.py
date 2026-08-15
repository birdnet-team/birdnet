from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
from multiprocessing import Queue
from multiprocessing.context import BaseContext
from multiprocessing.sharedctypes import Synchronized
from types import TracebackType

# How long a waiter blocks on a cross-process lock before looking up to see
# whether the run is still alive. Long enough that normal contention never
# reaches the check (the protected sections are microseconds), short enough that
# a leaked lock is noticed promptly.
LOCK_POLL_INTERVAL_S = 1.0


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


def acquire_or_give_up_when_cancelled(
  lock: multiprocessing.synchronize.Lock,
  cancel_event: multiprocessing.synchronize.Event,
  poll_interval_s: float = LOCK_POLL_INTERVAL_S,
) -> bool:
  """Take a cross-process lock, giving up if the run has been cancelled.

  A ``multiprocessing.Lock`` is a POSIX semaphore, and a semaphore held by a
  process that is killed -- SIGKILL from the OOM killer, a native crash -- is
  never posted again, so every other process blocking on it blocks for good.
  (Windows differs: a mutex owned by a dead process is *abandoned* and the next
  waiter acquires it, which is why the same run recovers there.) Blocking
  outright would therefore turn one dead child into a wedged pipeline that no
  liveness check can report, because the survivors are alive -- just stuck.

  Polling makes the leak survivable instead: the parent notices the dead child
  within a second and sets the cancel event, and every waiter here then leaves
  on its own. Returns ``False`` when it gave up, in which case the caller has
  *not* got the lock and must not touch what the lock protects.
  """
  while not lock.acquire(timeout=poll_interval_s):
    if cancel_event.is_set():
      return False
  return True


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
    self._counter_lock_lost = False

  def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
    acquired = self._sem.acquire(block, timeout)
    if acquired:
      self._adjust_counter(-1)
    return acquired

  def release(self) -> None:
    # Counter first, then the semaphore: a permit must never be visible to a
    # waiter before the gauge has caught up (see PipelineResources.reset).
    self._adjust_counter(+1)
    self._sem.release()

  def _adjust_counter(self, delta: int) -> None:
    """Move the mirrored counter without ever blocking indefinitely on it.

    The counter is only a gauge for the progress display -- the semaphore is
    what the pipeline actually runs on. Its lock is a POSIX semaphore all the
    same, so a process killed while holding it would otherwise stop every other
    process here for the rest of the run. Giving up costs a gauge that reads a
    few counts off; blocking would cost the run.
    """
    if self._counter_lock_lost:
      return
    lock = self._counter.get_lock()
    if not lock.acquire(timeout=LOCK_POLL_INTERVAL_S):
      # A microsecond-long critical section that cannot be entered within a
      # second is not contention, it is a lock that nobody will release again.
      # Stop trying, or every later call pays the same second.
      self._counter_lock_lost = True
      return
    try:
      self._counter.value += delta
    finally:
      lock.release()

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
