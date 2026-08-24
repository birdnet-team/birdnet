from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.synchronize
import queue as pyqueue
import threading
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.context import BaseContext
from multiprocessing.sharedctypes import Synchronized
from types import TracebackType
from typing import Any

_LOGGER = logging.getLogger(__name__)

# How long a waiter blocks on a cross-process lock before looking up to see
# whether the run is still alive. Long enough that normal contention never
# reaches the check (the protected sections are microseconds), short enough that
# a leaked lock is noticed promptly.
LOCK_POLL_INTERVAL_S = 1.0
# How long ThreadedQueueReader.close waits for its pump to stop. The pump polls
# every _PUMP_POLL_S, so a healthy one stops well inside this; one that does not
# is stuck inside the queue and will never stop.
READER_CLOSE_TIMEOUT_S = 2.0
# How long a message a finished child already promised may take to arrive. The
# sender handed it to its feeder thread before setting its finish signal, so on
# a healthy run it is milliseconds away; this bounds the child that was killed
# on its way out, which the liveness check can no longer flag because the
# signal is already set.
PROMISED_MESSAGE_DEADLINE_S = 30.0


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


@dataclass(frozen=True)
class _ReaderFailure:
  """A payload the pump could receive but not deserialize, forwarded as data."""

  error: Exception


class ThreadedQueueReader:
  """Reads a cross-process queue on its own thread, so the caller cannot block.

  ``Queue.get`` is unbounded in a way its signature hides: the timeout covers
  acquiring the read lock and the poll, but not ``_recv_bytes`` -- so a message
  a killed child left half-written stops the calling thread for good, with
  nothing raised (issues #77/#83). And the write side loops on partial writes,
  so this is not preventable by keeping messages small: any message can be cut
  mid-frame when the pipe is full, down to the platform's PIPE_BUF (512 bytes
  on macOS). What can be controlled is who gets stopped.

  Reading through a sacrificial daemon thread confines the stop: the caller
  reads an in-process buffer whose timeouts are real, its liveness checks keep
  running, and a torn frame costs a parked thread instead of the run.

  One reader per queue per run. ``close`` is bounded; a pump that does not stop
  is wedged inside the queue, and the owner must then leave that queue open --
  closing it would free its descriptor for reuse underneath the blocked read
  (see ProcessManager._collect_wedged_drainers).
  """

  # Also the latency ``close`` adds to every run: a healthy pump is usually
  # asleep in ``get(timeout=_PUMP_POLL_S)`` when close is called, and readers
  # are closed per run -- at 0.2 s this put a visible fixed barrier back into
  # warm ``run_arrays`` calls, which is the regression the session-overhead
  # test guards. Waiting is done by the *caller* on the buffer, never here, so
  # a short poll costs only a cheap uncontended wake per interval.
  _PUMP_POLL_S = 0.02

  # Restores the backpressure the pipe used to provide: without a bound the
  # pump would drain everything a fast backend produces straight into the
  # parent's heap. At the largest real block (~45 KB) this is ~11 MB, and a
  # full buffer parks the pump, which parks the pipe, which parks the writer
  # -- the same brake as before, without the unbounded read.
  _BUFFER_MAX_ITEMS = 256

  def __init__(self, q: Queue, name: str) -> None:
    self._q = q
    self._buffer: pyqueue.Queue = pyqueue.Queue(maxsize=self._BUFFER_MAX_ITEMS)
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._pump, name=name, daemon=True)
    self._thread.start()

  def _pump(self) -> None:
    while not self._stop.is_set():
      try:
        item = self._q.get(timeout=self._PUMP_POLL_S)
      except pyqueue.Empty:
        continue
      except Exception as e:  # noqa: BLE001
        # A payload that arrived intact but cannot be deserialized surfaces
        # here as any number of exceptions. Forward it instead of dying
        # silently: the caller then fails the run the same way it would have
        # when it read the queue itself, rather than waiting on a buffer
        # nobody fills any more.
        self._forward(_ReaderFailure(e))
        return
      if not self._forward(item):
        return

  def _forward(self, item: object) -> bool:
    # Never an unconditional put: with the buffer full that would block, and
    # close() would then misreport a merely-parked pump as wedged.
    while not self._stop.is_set():
      try:
        self._buffer.put(item, timeout=self._PUMP_POLL_S)
        return True
      except pyqueue.Full:
        continue
    return False

  def _unwrap(self, item: object) -> object:
    if isinstance(item, _ReaderFailure):
      raise RuntimeError(
        "A message on a pipeline queue could not be deserialized."
      ) from item.error
    return item

  def get(self, timeout: float) -> Any:
    """Bounded for real, unlike ``Queue.get``. Raises ``queue.Empty``."""
    return self._unwrap(self._buffer.get(timeout=timeout))

  def get_nowait(self) -> Any:
    return self._unwrap(self._buffer.get_nowait())

  def close(self, timeout: float = READER_CLOSE_TIMEOUT_S) -> bool:
    """Stop the pump; ``False`` means it is wedged inside the queue."""
    self._stop.set()
    self._thread.join(timeout=timeout)
    return not self._thread.is_alive()


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
