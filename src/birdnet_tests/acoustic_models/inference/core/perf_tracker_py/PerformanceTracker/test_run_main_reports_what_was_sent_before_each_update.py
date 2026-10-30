"""Every progress update reports the stats the workers had sent when it was built.

`run_main` waits an interval between updates, and the workers report during
that wait; the stats have to be collected after it, or every update is one
interval behind. The waits are simulated, not slept: the stand-in for the
end-of-run semaphore reports worker stats each time the tracker waits on it.

The closing update too: a worker puts its last result before its stats, so the
end-of-run token can reach the tracker first, and a short count keeps the
tracker draining for up to one interval before it closes. Those pauses run on
a stand-in clock that skips ahead instead of sleeping.
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import birdnet.acoustic.inference.core.perf_tracker as perf_tracker_module
from birdnet.acoustic.inference.core.perf_tracker import (
  AcousticProgressStats,
  PerformanceTracker,
)
from birdnet.globals import WRITABLE_FLAG

pytestmark = pytest.mark.no_tf


class _Workers(queue.Queue[Any]):
  """The worker stats queue, filled with single-segment batches as workers do.

  Batches held back reach the queue just before a later drain of the tracker's,
  the way a worker's stats can trail the end-of-run token.
  """

  def __init__(self) -> None:
    super().__init__()
    self.reported = 0
    self.drains = 0
    self._held_back = 0
    self._release_on_drain = 0

  def report(self, n_batches: int) -> None:
    for _ in range(n_batches):
      self.reported += 1
      wall_time_s = 0.25 * self.reported
      super().put((4242, wall_time_s, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 1))

  def hold_back(self, n_batches: int, drains_from_now: int) -> None:
    self._held_back = n_batches
    self._release_on_drain = self.drains + drains_from_now

  def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
    # The tracker's end marker: a drain starts.
    if item is None:
      self.drains += 1
      if self.drains == self._release_on_drain:
        self.report(self._held_back)
        self._held_back = 0
    super().put(item, block, timeout)


class _EndOfRun:
  """Stands in for the semaphore the session releases once the last result is in.

  Each wait lets the workers report the given number of batches; the last wait
  is the one that gets the token. Of that wait's batches, the last `trailing`
  report `drains_late` drains after the drain that follows the token. A cancel
  during wait `cancel_on_wait` arrives as the parent's does: set while the
  tracker waits, so the wait returns without the token.
  """

  def __init__(
    self,
    workers: _Workers,
    batches_per_wait: list[int],
    trailing: int,
    drains_late: int,
    cancel_event: threading.Event,
    cancel_on_wait: int | None,
  ) -> None:
    self._workers = workers
    self._batches_per_wait = batches_per_wait
    self._trailing = trailing
    self._drains_late = drains_late
    self._cancel_event = cancel_event
    self._cancel_on_wait = cancel_on_wait
    self.waits = 0

  def acquire(self, timeout: float) -> bool:
    assert self.waits < len(self._batches_per_wait), "waited after the token"
    n_batches = self._batches_per_wait[self.waits]
    self.waits += 1
    token = self.waits == len(self._batches_per_wait)
    if token and self._trailing > 0:
      self._workers.hold_back(self._trailing, 1 + self._drains_late)
      n_batches -= self._trailing
    self._workers.report(n_batches)
    if self.waits == self._cancel_on_wait:
      self._cancel_event.set()
      return False
    return token


class _Updates:
  """The callback queue: records each update with what had been reported by then."""

  def __init__(self, reported: Callable[[], int]) -> None:
    self._reported = reported
    self.sent: list[tuple[AcousticProgressStats, int]] = []

  def put_nowait(self, stats: AcousticProgressStats) -> None:
    self.sent.append((stats, self._reported()))


class _Clock:
  """The tracker's clock: real time plus every sleep, which returns at once."""

  def __init__(self, on_sleep: Callable[[], None]) -> None:
    self.slept: list[float] = []
    self._on_sleep = on_sleep

  def perf_counter(self) -> float:
    return time.perf_counter() + sum(self.slept)

  def sleep(self, seconds: float) -> None:
    if seconds < 0:
      raise ValueError("sleep length must be non-negative")
    self.slept.append(seconds)
    assert len(self.slept) < 10_000, "the closing window never ended"
    self._on_sleep()


class _Gauge:
  def get_value(self) -> int:
    return 0


@dataclass
class _Run:
  updates: _Updates
  end_of_run: _EndOfRun
  workers: _Workers
  clock: _Clock
  perf_res: queue.Queue[Any]


def _run(
  monkeypatch: pytest.MonkeyPatch,
  batches_per_wait: list[int],
  *,
  trailing: int = 0,
  drains_late: int = 0,
  total: int | None = None,
  update_interval: float = 1.0,
  cancel_on_sleep: bool = False,
  cancel_on_wait: int | None = None,
) -> _Run:
  workers = _Workers()
  cancel_event = threading.Event()
  end_of_run = _EndOfRun(
    workers, batches_per_wait, trailing, drains_late, cancel_event, cancel_on_wait
  )
  updates = _Updates(lambda: workers.reported)
  clock = _Clock(cancel_event.set if cancel_on_sleep else lambda: None)
  monkeypatch.setattr(perf_tracker_module, "time", clock)
  perf_res: queue.Queue[Any] = queue.Queue()
  # What the session would hand over; only the parts run_main touches act.
  stand_ins: dict[str, Any] = {
    "pred_dur_queue": workers,
    "prod_stats_queue": queue.Queue(),
    "callback_queue": updates,
    "processing_finished": end_of_run,
    "logging_queue": queue.Queue(),
    "perf_res": perf_res,
    "sem_active_workers": _Gauge(),
    "sem_filled_slots": _Gauge(),
    "rf_flags": None,
    "start_signal": None,
  }
  tracker = PerformanceTracker(
    session_id="test-run-main-freshness",
    update_interval=update_interval,
    n_workers=1,
    logging_level=logging.DEBUG,
    segment_size_s=3.0,
    parent_process_id=os.getpid(),
    tot_n_segments_ptr=ctypes.c_uint64(
      sum(batches_per_wait) if total is None else total
    ),
    cancel_event=cancel_event,
    end_event=threading.Event(),
    finish_signal=threading.Event(),
    start=time.perf_counter(),
    # Not "fork": the constructor then attaches neither logging nor ring flags.
    start_method="spawn",
    **stand_ins,
  )
  tracker._log = lambda *_: None  # type: ignore[method-assign]  # noqa: SLF001
  tracker._ring_flags = np.full(4, WRITABLE_FLAG)  # noqa: SLF001

  tracker.run_main()

  return _Run(updates, end_of_run, workers, clock, perf_res)


def _assert_each_update_is_current(updates: _Updates) -> None:
  for i, (stats, reported) in enumerate(updates.sent):
    assert stats.processed_segments == reported, (
      f"update {i} reports {stats.processed_segments} segments, but the workers "
      f"had sent {reported} by the time it was built"
    )


def _assert_only_the_last_is_finished(updates: _Updates) -> AcousticProgressStats:
  assert updates.sent, "no update was sent"
  assert not any(stats.finished for stats, _ in updates.sent[:-1])
  last, _ = updates.sent[-1]
  assert last.finished
  return last


def test_each_update_reports_the_batches_of_the_wait_before_it(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  run = _run(monkeypatch, [1, 1, 1, 1])

  _assert_each_update_is_current(run.updates)
  last = _assert_only_the_last_is_finished(run.updates)
  assert last.processed_segments == 4


def test_a_run_that_ends_during_the_first_wait_closes_at_100_percent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  run = _run(monkeypatch, [4])

  assert run.end_of_run.waits == 1, "a pass after the token waited again"
  _assert_each_update_is_current(run.updates)
  last = _assert_only_the_last_is_finished(run.updates)
  assert last.processed_segments == 4
  assert last.progress_pct == 100.0


@pytest.mark.parametrize("drains_late", [1, 3])
def test_stats_that_trail_the_token_reach_the_closing_update(
  monkeypatch: pytest.MonkeyPatch, drains_late: int
) -> None:
  run = _run(monkeypatch, [1, 1, 1, 1], trailing=1, drains_late=drains_late)

  _assert_each_update_is_current(run.updates)
  last = _assert_only_the_last_is_finished(run.updates)
  assert last.processed_segments == 4, (
    "the closing update missed the stats that trailed the token"
  )
  assert last.progress_pct == 100.0
  assert len(run.clock.slept) == drains_late, (
    f"paused {len(run.clock.slept)} times for stats {drains_late} drains late"
  )


@pytest.mark.parametrize(
  ("batches_per_wait", "drains"),
  [
    pytest.param([1, 1, 1, 0], 4, id="the-drain-after-the-token-finds-nothing"),
    pytest.param([1, 1, 1, 1], 5, id="the-drain-after-the-token-finds-the-last"),
  ],
)
def test_a_close_with_the_full_count_pays_nothing(
  monkeypatch: pytest.MonkeyPatch, batches_per_wait: list[int], drains: int
) -> None:
  run = _run(monkeypatch, batches_per_wait)

  assert run.clock.slept == [], "a close with the full count paused"
  assert run.workers.drains == drains, (
    f"{run.workers.drains} drains, expected {drains}: the close drained again"
  )
  last = _assert_only_the_last_is_finished(run.updates)
  assert last.processed_segments == sum(batches_per_wait)
  assert last.progress_pct == 100.0


def test_a_total_never_reached_closes_after_one_interval(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Six segments expected, four delivered: inputs a producer could not read
  # stay in the total.
  interval = 0.2
  run = _run(monkeypatch, [1, 1, 1, 1], total=6, update_interval=interval)

  last = _assert_only_the_last_is_finished(run.updates)
  assert last.processed_segments == 4
  assert last.total_segments == 6
  assert run.clock.slept, "the close did not wait for the missing segments"
  assert sum(run.clock.slept) <= interval + 1e-9, (
    f"paused {sum(run.clock.slept):.3f} s in a window of {interval} s"
  )
  before_last, _ = run.updates.sent[-2]
  assert last.wall_time_s - before_last.wall_time_s >= interval, (
    "the close came before the window of one interval ran out"
  )


def test_a_cancel_during_the_closing_window_stops_before_the_next_drain(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # After a cancel the parent drains the stats queues too; a tracker drain then
  # could lose its end marker to it and block for good.
  run = _run(monkeypatch, [1, 1], total=4, cancel_on_sleep=True)

  assert run.workers.drains == 3, "the tracker drained after the cancel"
  assert len(run.clock.slept) == 1
  assert not any(stats.finished for stats, _ in run.updates.sent)
  assert run.perf_res.empty()


def test_a_cancel_during_the_wait_stops_before_the_drain(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # The same hazard, at the check right after the wait.
  run = _run(monkeypatch, [1, 1, 1], cancel_on_wait=2)

  assert run.end_of_run.waits == 2
  assert run.workers.drains == 1, "the tracker drained after the cancel"
  assert not any(stats.finished for stats, _ in run.updates.sent)
  assert run.perf_res.empty()


def test_the_closing_window_samples_live_stats_once_per_pass(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Live samples (CPU, memory, ring slots) feed the run's averages; the window's
  # drains, 50 a second, must not add to them.
  live_samples = 0
  original = PerformanceTracker._track_live_stats  # noqa: SLF001

  def counting(self: PerformanceTracker) -> None:
    nonlocal live_samples
    live_samples += 1
    original(self)

  monkeypatch.setattr(PerformanceTracker, "_track_live_stats", counting)
  batches_per_wait = [1, 1, 1, 1]

  run = _run(monkeypatch, batches_per_wait, trailing=1, drains_late=3)

  assert len(run.clock.slept) == 3, "the window did not run"
  assert live_samples == len(batches_per_wait), (
    f"{live_samples} live samples for {len(batches_per_wait)} passes: the "
    f"window sampled them too"
  )
