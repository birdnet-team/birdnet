"""The slot gauge must never be able to stop the pipeline it only reports on.

`CountedSemaphore` mirrors a semaphore's count into a shared value so
`get_value()` works on macOS, where `sem_getvalue` is unimplemented. That mirror
has a lock of its own, and a lock is a semaphore: a process killed while holding
it never releases it, so waiting on it would stop every producer and worker
handing a slot over -- for a number that only feeds the progress display.

The lock behind `multiprocessing.Value` is an *RLock*, so a test that takes it
on the same thread it then measures gets it back re-entrantly and exercises
nothing. The leaked lock has to be held by another thread to stand in for the
process that died holding it.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from birdnet.acoustic.inference.core.sync import CountedSemaphore

_SYNC_LOGGER = "birdnet.acoustic.inference.core.sync"
# The give-up costs one poll interval (1 s) the first time and nothing after.
# These bound "did not wait again", not how fast a machine is.
_ONE_GIVE_UP_S = 10.0
_AFTER_THE_LATCH_S = 2.0
_HOLDER_DEADLINE_S = 30.0
# Reading the gauge is a memory load; this only separates it from a lock wait.
_READ_DEADLINE_S = 5.0
# The holder lets go on its own after this even if nobody asks it to. Without
# that, removing the give-up would make these tests *hang* inside the `with`
# rather than fail: the release in the exit handler is unreachable while the
# call under test is still blocked, and the assertion would never be evaluated.
_HOLD_CEILING_S = _ONE_GIVE_UP_S * 2


@pytest.fixture
def warnings_from_sync() -> Iterator[list[logging.LogRecord]]:
  """Records straight off the module logger.

  Not caplog: the package logger sets `propagate = False`, so records never
  reach the root handler caplog installs.
  """
  records: list[logging.LogRecord] = []

  class _Collect(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
      records.append(record)

  logger = logging.getLogger(_SYNC_LOGGER)
  handler = _Collect()
  previous_level = logger.level
  logger.setLevel(logging.WARNING)
  logger.addHandler(handler)
  try:
    yield records
  finally:
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


def _semaphore() -> CountedSemaphore:
  return CountedSemaphore(0, ctx=mp.get_context("spawn"))


@contextmanager
def _held_by_another_thread(lock) -> Iterator[threading.Event]:  # noqa: ANN001
  """Hold `lock` from a thread that will not give it back on its own.

  Stands in for the process killed while holding it. Yields the event that
  finally releases it, so a test can hand the lock back mid-way to check that
  ordinary contention still resolves.
  """
  held = threading.Event()
  release = threading.Event()

  def hold() -> None:
    lock.acquire()
    held.set()
    release.wait(_HOLD_CEILING_S)
    lock.release()

  holder = threading.Thread(target=hold, name="counter-lock-holder", daemon=True)
  holder.start()
  assert held.wait(timeout=_HOLDER_DEADLINE_S), "the holder never took the lock"
  try:
    yield release
  finally:
    release.set()
    holder.join(timeout=_HOLDER_DEADLINE_S)


def test_the_gauge_tracks_the_semaphore_while_the_lock_is_healthy() -> None:
  sem = _semaphore()

  sem.release()
  sem.release()
  assert sem.get_value() == 2

  assert sem.acquire(block=False)
  assert sem.get_value() == 1


def test_a_counter_lock_nobody_releases_does_not_stop_the_semaphore(
  warnings_from_sync: list[logging.LogRecord],
) -> None:
  """The regression: without the give-up, `release` never returns."""
  sem = _semaphore()

  with _held_by_another_thread(sem._counter.get_lock()):
    start = time.monotonic()
    sem.release()
    elapsed_s = time.monotonic() - start

    assert elapsed_s < _ONE_GIVE_UP_S, (
      f"release() took {elapsed_s:.1f} s: it waited on a lock nobody will release"
    )
    # The permit is what the pipeline actually runs on, so it must have been
    # handed over even though the gauge could not be updated.
    assert sem.acquire(block=False), "the permit was lost when the gauge gave up"
    # ...and the gauge is the part that gives: it saw neither operation.
    assert sem.get_value() == 0

  assert any("gauges are frozen" in r.getMessage() for r in warnings_from_sync), (
    "giving up on the gauge must be reported; nothing else would show it"
  )


def test_reading_the_gauge_does_not_wait_on_the_counter_lock() -> None:
  """Reading it must not block either, or the fix only moves the block.

  `Synchronized.value` is a property that takes the counter lock, and the
  performance tracker reads this on every stats interval -- so a killed process
  holding that lock would stop the tracker instead of the workers, and the
  parent would have to terminate it.
  """
  sem = _semaphore()

  with _held_by_another_thread(sem._counter.get_lock()):
    start = time.monotonic()
    value = sem.get_value()
    elapsed_s = time.monotonic() - start

  assert elapsed_s < _READ_DEADLINE_S, (
    f"get_value() took {elapsed_s:.1f} s: it waited on the counter lock"
  )
  assert value == 0


def test_the_counter_lock_is_not_waited_on_a_second_time(
  warnings_from_sync: list[logging.LogRecord],
) -> None:
  """Each later call would otherwise pay the same poll interval again."""
  sem = _semaphore()

  with _held_by_another_thread(sem._counter.get_lock()):
    sem.release()  # pays the one poll interval and latches

    start = time.monotonic()
    for _ in range(3):
      sem.release()
    elapsed_s = time.monotonic() - start

  assert elapsed_s < _AFTER_THE_LATCH_S, (
    f"three releases took {elapsed_s:.1f} s; the give-up is not remembered, so "
    f"every call pays the poll interval again"
  )
  assert len(warnings_from_sync) == 1, "the warning must be said once, not per call"


def test_ordinary_contention_is_not_mistaken_for_a_leak() -> None:
  """A lock held briefly by someone else must still be waited for."""
  sem = _semaphore()

  with _held_by_another_thread(sem._counter.get_lock()) as release:
    handover = threading.Timer(0.2, release.set)
    handover.daemon = True
    handover.start()
    try:
      sem.release()
    finally:
      handover.cancel()

  assert sem.get_value() == 1, "the gauge was abandoned over ordinary contention"
