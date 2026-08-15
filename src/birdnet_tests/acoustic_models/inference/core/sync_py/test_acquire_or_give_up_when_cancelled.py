"""A leaked cross-process lock must end the run, not wedge every waiter.

`multiprocessing.Lock` is a semaphore on every platform -- POSIX or Windows --
and a semaphore has no owner to abandon it to the next waiter. A process killed
while holding one never posts it again, so every other process blocking on it
blocks forever. The parent's liveness check reports the child that died, but
the survivors are alive -- just stuck -- so nothing else can free them
(issue #73). Waiting in bounded steps and giving up when the run is cancelled
is what makes the leak survivable.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

from birdnet.acoustic.inference.core.sync import acquire_or_give_up_when_cancelled

# Bounds a wait that must not happen at all, not a duration to measure.
_DEADLINE_S = 30.0


def test_an_uncontended_lock_is_taken() -> None:
  ctx = mp.get_context("spawn")
  lock = ctx.Lock()
  cancel = ctx.Event()

  assert acquire_or_give_up_when_cancelled(lock, cancel, poll_interval_s=0.05)

  lock.release()


def test_a_never_released_lock_is_abandoned_once_the_run_is_cancelled() -> None:
  """The regression: without the give-up this call never returns."""
  ctx = mp.get_context("spawn")
  lock = ctx.Lock()
  cancel = ctx.Event()
  # Stands in for the killed holder: on POSIX its semaphore is never posted.
  lock.acquire()
  cancel.set()

  done = threading.Event()
  result: dict[str, bool] = {}

  def waiter() -> None:
    result["got_lock"] = acquire_or_give_up_when_cancelled(
      lock, cancel, poll_interval_s=0.05
    )
    done.set()

  threading.Thread(target=waiter, name="ring-lock-waiter", daemon=True).start()

  assert done.wait(timeout=_DEADLINE_S), (
    "the waiter never gave up on a lock that is never released"
  )
  assert result["got_lock"] is False, (
    "giving up must be reported, or the caller would touch the ring buffer "
    "without holding the lock"
  )

  lock.release()


def test_it_keeps_waiting_while_the_run_is_healthy() -> None:
  """Ordinary contention must not be mistaken for a leak."""
  ctx = mp.get_context("spawn")
  lock = ctx.Lock()
  cancel = ctx.Event()
  lock.acquire()

  done = threading.Event()
  result: dict[str, bool] = {}

  def waiter() -> None:
    result["got_lock"] = acquire_or_give_up_when_cancelled(
      lock, cancel, poll_interval_s=0.05
    )
    done.set()

  threading.Thread(target=waiter, name="ring-lock-waiter", daemon=True).start()

  # Long enough for several poll intervals to pass: the waiter must still be
  # waiting, because nothing told it the run was over.
  assert not done.wait(timeout=0.5), "gave up on a healthy run"

  lock.release()

  assert done.wait(timeout=_DEADLINE_S)
  assert result["got_lock"] is True


def test_the_lock_is_handed_over_rather_than_polled_away() -> None:
  """It must actually acquire, not merely notice the lock became free."""
  ctx = mp.get_context("spawn")
  lock = ctx.Lock()
  cancel = ctx.Event()
  lock.acquire()

  acquired = threading.Event()

  def waiter() -> None:
    if acquire_or_give_up_when_cancelled(lock, cancel, poll_interval_s=0.05):
      acquired.set()

  threading.Thread(target=waiter, name="ring-lock-waiter", daemon=True).start()
  time.sleep(0.1)
  lock.release()

  assert acquired.wait(timeout=_DEADLINE_S)
  # Still held by the waiter, so a second non-blocking attempt must fail.
  assert not lock.acquire(block=False), "the lock was not actually held"
