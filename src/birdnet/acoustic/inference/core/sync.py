from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.synchronize
from multiprocessing import Queue
from multiprocessing.context import BaseContext
from multiprocessing.sharedctypes import Synchronized
from types import TracebackType

_LOGGER = logging.getLogger(__name__)

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

  A ``multiprocessing.Lock`` is a semaphore on every platform -- a POSIX
  semaphore, or a Windows kernel semaphore (``CreateSemaphore``), not a mutex.
  Neither is released when its holder is killed by SIGKILL from the OOM killer
  or by a native crash, and semaphores have no owner to abandon them to the
  next waiter, so every other process blocking on one blocks for good. Blocking
  outright would therefore turn one dead child into a wedged pipeline that no
  liveness check can report, because the survivors are alive -- just stuck.

  Polling makes the leak survivable instead: the parent notices the dead child
  within a second and sets the cancel event, and every waiter here then leaves
  on its own. Returns ``False`` when it gave up, in which case the caller has
  *not* got the lock and must not touch what the lock protects.

  The non-blocking attempt first is not an optimization for its own sake. macOS
  has no ``sem_timedwait``, so CPython emulates *any* timed acquire with a
  poll loop that sleeps in steps of up to 20 ms -- which would put that latency
  on a per-batch path, and replace the kernel's wait queue with a free-for-all
  that can starve a waiter. ``acquire(block=False)`` is a plain ``sem_trywait``
  on every platform, so an uncontended take -- effectively all of them, the
  protected sections being microseconds -- costs exactly what it did before,
  and only a genuinely contended one falls back to polling.
  """
  if lock.acquire(block=False):
    return True
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
    what the pipeline actually runs on. Its lock is a semaphore all the same,
    so a process killed while holding it would otherwise stop every other
    process here for the rest of the run. Giving up costs a gauge that reads a
    few counts off from then on; blocking would cost the run.
    """
    if self._counter_lock_lost:
      return
    lock = self._counter.get_lock()
    # Non-blocking first, for the reason given in
    # acquire_or_give_up_when_cancelled: this runs on every permit handed
    # between a producer and a worker, and a timed acquire is a poll loop on
    # macOS.
    if not lock.acquire(block=False) and not lock.acquire(timeout=LOCK_POLL_INTERVAL_S):
      # A microsecond-long critical section that cannot be entered within a
      # second is not contention, it is a lock that nobody will release again.
      # Stop trying, or every later call pays the same second. Said out loud
      # because the gauges silently freeze from here on, and a reader of the
      # progress output has no other way to know.
      self._counter_lock_lost = True
      _LOGGER.warning(
        "Gave up on the slot-counter lock; it is held by a process that will "
        "not release it. The slot and busy-worker gauges are frozen for the "
        "rest of this process."
      )
      return
    try:
      self._counter.value += delta
    finally:
      lock.release()

  def get_value(self) -> int:
    # Deliberately not ``self._counter.value``: that property takes the counter
    # lock, which is exactly the lock a killed process leaves held -- and this
    # is read by the performance tracker on every stats interval, so it would
    # simply move the block from the writers to the reader. A gauge does not
    # need the lock: reads of a C int do not tear, and a count that is one
    # behind costs a progress line nothing.
    return self._counter.get_obj().value

  def __enter__(self) -> bool:
    return self.acquire()

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: TracebackType | None,
  ) -> None:
    self.release()
