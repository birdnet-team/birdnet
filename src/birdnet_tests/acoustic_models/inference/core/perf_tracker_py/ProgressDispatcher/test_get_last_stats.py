"""`get_last_stats` hands an update over as it arrives, the newest of a burst.

It waits up to one check interval for the first update and then takes only
what is already queued, so an update is not held back for a quiet interval
after it arrives.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Iterator
from typing import cast

import pytest

from birdnet.acoustic.inference.core.perf_tracker import (
  AcousticProgressStats,
  BufferStats,
  ProducerStats,
  ProgressDispatcher,
)
from birdnet.acoustic.inference.core.sync import ChildToParentQueue

pytestmark = pytest.mark.no_tf

# Long, so a drain that waits out a quiet interval cannot pass for a prompt one.
_CHECK_INTERVAL_S = 10.0
# A lone update is milliseconds away; this catches a drain that still waits a
# fraction of the interval, while leaving a loaded runner plenty of room.
_LONE_UPDATE_BOUND_S = 1.0


def _update(processed_segments: int) -> AcousticProgressStats:
  return AcousticProgressStats(
    finished=False,
    buffer_stats=BufferStats(slots=1, free_slots=1, busy_slots=0, preloaded_slots=0),
    producer_stats=ProducerStats(
      speed_xrt=0, speed_seg_per_s=0, wait_ms=0, batch_ms=0, search_ms=0, flush_ms=0
    ),
    worker_stats=None,
    wall_time_s=0.0,
    memory_usage_MiB=0.0,
    memory_usage_max_MiB=0.0,
    cpu_usage_pct=0.0,
    cpu_usage_max_pct=0.0,
    progress_pct=0.0,
    est_remaining_time_s=None,
    processed_segments=processed_segments,
    processed_batches=processed_segments,
    total_segments=None,
    speed_xrt=None,
    speed_seg_per_s=None,
  )


def _dispatcher(q: object, check_interval_s: float) -> ProgressDispatcher:
  ctx = mp.get_context("spawn")
  return ProgressDispatcher(
    session_id="test-get-last-stats",
    callback_queue=cast(ChildToParentQueue, q),
    callback_fn=lambda _: None,
    cancel_event=ctx.Event(),
    end_event=ctx.Event(),
    start_signal=threading.Event(),
    finish_signal=threading.Event(),
    last_update_signal=threading.Event(),
    busy_signal=threading.Event(),
    processing_finished_event=threading.Event(),
    check_interval=check_interval_s,
  )


@pytest.fixture(params=["multiprocessing", "thread"])
def any_queue(request: pytest.FixtureRequest) -> Iterator[object]:
  if request.param == "thread":
    yield queue.Queue()
    return
  q = mp.get_context("spawn").Queue()
  try:
    yield q
  finally:
    q.close()
    q.cancel_join_thread()


def test_a_lone_update_is_returned_as_it_arrives(any_queue: object) -> None:
  dispatcher = _dispatcher(any_queue, _CHECK_INTERVAL_S)
  update = _update(1)

  start = time.monotonic()
  cast(ChildToParentQueue, any_queue).put(update)
  latest = dispatcher.get_last_stats()
  elapsed_s = time.monotonic() - start

  assert latest == update
  assert elapsed_s < _LONE_UPDATE_BOUND_S, (
    f"the update took {elapsed_s:.1f} s to come back: the drain waited for a "
    f"quiet interval of {_CHECK_INTERVAL_S:.0f} s after it"
  )


def test_a_burst_returns_its_newest_update() -> None:
  # A thread queue's put is synchronous, so the whole burst is queued before
  # the drain starts; a multiprocessing queue's feeder thread gives no such
  # point.
  q: queue.Queue[AcousticProgressStats] = queue.Queue()
  dispatcher = _dispatcher(q, _CHECK_INTERVAL_S)
  for i in range(1, 6):
    q.put(_update(i))

  latest = dispatcher.get_last_stats()

  assert latest is not None
  assert latest.processed_segments == 5
  assert q.empty()


def test_nothing_queued_returns_none_after_one_interval() -> None:
  check_interval_s = 0.2
  dispatcher = _dispatcher(queue.Queue(), check_interval_s)

  start = time.monotonic()
  latest = dispatcher.get_last_stats()
  elapsed_s = time.monotonic() - start

  assert latest is None
  assert elapsed_s >= check_interval_s, "it gave up before the interval was over"
