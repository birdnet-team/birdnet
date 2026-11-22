"""A finish signal: a child sets it, the parent polls it and waits on it.

The parent waits on it for real, so a finished child ends the wait at once and
not at the next poll, and asks again after the run, so a set signal keeps
answering at once. The child's ``set()`` must return whoever died inside a
wait: ``Event.set()`` waits for every waiter to confirm it woke, and a killed
one never does.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
import threading
import time

import pytest

from birdnet.acoustic.inference.core.sync import FinishSignal
from birdnet.core.start_method import resolve_start_method

pytestmark = pytest.mark.no_tf

# A spawned interpreter's start, longer on a loaded runner.
_START_S = 120.0
# Far longer than any healthy wake, and the bound below stays far from it: a
# wait that only its timeout ends is told apart from one the signal ended.
_WAIT_TIMEOUT_S = 60.0
_WAKE_BOUND_S = 20.0


def _context() -> mp.context.BaseContext:
  return mp.get_context(resolve_start_method())


def _set_after(
  signal: FinishSignal, started: multiprocessing.synchronize.Semaphore, delay_s: float
) -> None:
  started.release()
  time.sleep(delay_s)
  signal.set()


def _wait_for(
  signal: FinishSignal, waiting: multiprocessing.synchronize.Semaphore
) -> None:
  waiting.release()
  signal.wait(timeout=3600.0)


def test_a_set_from_another_process_ends_the_wait() -> None:
  ctx = _context()
  signal = FinishSignal(ctx)
  started = ctx.Semaphore(0)
  child = ctx.Process(target=_set_after, args=(signal, started, 0.2))
  child.start()
  try:
    assert started.acquire(timeout=_START_S)
    begin = time.monotonic()

    assert signal.wait(timeout=_WAIT_TIMEOUT_S)

    elapsed = time.monotonic() - begin
    assert elapsed < _WAKE_BOUND_S, (
      f"the wait ended after {elapsed:.1f} s: the set did not wake it"
    )
  finally:
    child.join(timeout=_START_S)
  assert child.exitcode == 0


def test_a_set_signal_answers_every_later_wait_at_once() -> None:
  """The parent asks again after the run; the wake was spent by the first wait."""
  signal = FinishSignal(_context())
  signal.set()
  assert signal.wait(timeout=_WAIT_TIMEOUT_S)

  for _ in range(3):
    begin = time.monotonic()
    assert signal.wait(timeout=_WAIT_TIMEOUT_S)
    assert time.monotonic() - begin < _WAKE_BOUND_S, "a set signal was waited on"
  assert signal.is_set()


def test_a_cleared_signal_waits_out_its_timeout() -> None:
  """``reset()`` between runs: last run's wake must not end this run's wait."""
  signal = FinishSignal(_context())
  signal.set()
  signal.clear()
  timeout_s = 0.5
  begin = time.monotonic()

  assert not signal.wait(timeout=timeout_s)

  # Less a margin for clocks that tick coarser than the timeout's own.
  assert time.monotonic() - begin >= timeout_s - 0.1, "a stale wake ended the wait"
  assert not signal.is_set()


def test_a_set_whose_waiter_was_killed_inside_its_wait_returns() -> None:
  ctx = _context()
  signal = FinishSignal(ctx)
  waiting = ctx.Semaphore(0)
  waiter = ctx.Process(target=_wait_for, args=(signal, waiting), daemon=True)
  waiter.start()
  try:
    assert waiting.acquire(timeout=_START_S)
    # Long enough that the waiter is inside wait(), not on its way there.
    time.sleep(0.5)
  finally:
    waiter.kill()
    waiter.join()

  returned = threading.Event()

  def set_signal() -> None:
    signal.set()
    returned.set()

  # A thread, so a set() that never returns fails the test instead of hanging it.
  threading.Thread(target=set_signal, name="finish-set", daemon=True).start()

  assert returned.wait(timeout=_WAKE_BOUND_S), "set() blocked on a dead waiter"
  assert signal.wait(timeout=_WAIT_TIMEOUT_S)
