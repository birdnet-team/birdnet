"""A worker whose wait times out as the last batch is published still reads it.

The last producer publishes its final batch and then sets all-finished. Both
can land between a worker's timed wait running out and its look at the flag;
the worker then finds the flag set with the batch's permit still there, and if
no other worker is waiting, nobody else will read that batch. So before it
leaves, it takes a permit without blocking. The cancel check stays first.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from collections.abc import Callable
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from birdnet.acoustic.inference.core.worker import WorkerBase
from birdnet.globals import READABLE_FLAG, WRITABLE_FLAG

pytestmark = pytest.mark.no_tf

_N_SLOTS = 2
_BATCH_SIZE = 2
_SAMPLES = 3


class _ScriptedFilled:
  """Ring permits whose timed waits are scripted, so no test waits out a poll.

  The first timed wait has expired when ``on_first_timeout`` runs: a permit it
  releases is one the wait did not see. Every later timed wait takes a permit
  if there is one and otherwise expires at once.
  """

  def __init__(self, on_first_timeout: Callable[[], None]) -> None:
    self.permits = 0
    self.timed_waits = 0
    self._on_first_timeout = on_first_timeout

  def release(self) -> None:
    self.permits += 1

  def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
    if block:
      assert timeout is not None, "an unbounded wait for a ring permit"
      self.timed_waits += 1
      if self.timed_waits == 1:
        self._on_first_timeout()
        return False
    if self.permits > 0:
      self.permits -= 1
      return True
    return False


class _Out:
  def __init__(self) -> None:
    self.items: list[object] = []

  def put(self, obj: object, block: bool = True, timeout: float | None = None) -> None:
    self.items.append(obj)


def _worker(on_first_timeout: Callable[[SimpleNamespace], None]) -> SimpleNamespace:
  stub = SimpleNamespace(
    _backend=SimpleNamespace(copy_to_device=lambda batch: batch),
    _n_slots=_N_SLOTS,
    _ring_flags=np.full(_N_SLOTS, WRITABLE_FLAG, dtype=np.uint8),
    _ring_file_indices=np.zeros((_N_SLOTS, _BATCH_SIZE), dtype=np.uint8),
    _ring_segment_indices=np.zeros((_N_SLOTS, _BATCH_SIZE), dtype=np.uint16),
    _ring_audio_samples=np.zeros((_N_SLOTS, _BATCH_SIZE, _SAMPLES), dtype=np.float32),
    _ring_batch_sizes=np.zeros(_N_SLOTS, dtype=np.uint8),
    _wkr_ring_access_lock=mp.get_context("spawn").Lock(),
    _sem_free=threading.Semaphore(0),
    _sem_active_workers=None,
    _wkr_stats_queue=None,
    _out_q=_Out(),
    _cancel_event=threading.Event(),
    _all_producers_finished=threading.Event(),
    _parent_is_gone=lambda: False,
    _prediction_count=0,
    _log=lambda *args: None,
    _infer=lambda batch: batch,
    _result_to_host=lambda result: np.asarray(result),
    _get_block=lambda files, segments, result: (files, segments, result),
  )
  stub._sem_filled = _ScriptedFilled(lambda: on_first_timeout(stub))
  stub._check_cancel_event = MethodType(WorkerBase._check_cancel_event, stub)
  stub._take_filled_slot = MethodType(WorkerBase._take_filled_slot, stub)
  return stub


def _publish_last_batch_and_finish(worker: SimpleNamespace) -> None:
  """What the last producer does: publish, set all-finished, hand off a permit."""
  slot = 1
  worker._ring_file_indices[slot] = [0, 0]
  worker._ring_segment_indices[slot] = [6, 7]
  worker._ring_audio_samples[slot] = 0.5
  worker._ring_batch_sizes[slot] = _BATCH_SIZE
  worker._ring_flags[slot] = READABLE_FLAG
  worker._sem_filled.release()
  worker._all_producers_finished.set()
  worker._sem_filled.release()


def test_a_batch_published_as_the_wait_times_out_is_read_before_leaving() -> None:
  worker = _worker(_publish_last_batch_and_finish)

  WorkerBase.run_main(worker)

  items = worker._out_q.items
  assert items[-1] is None, f"no end-of-work sentinel last: {items}"
  blocks = items[:-1]
  assert len(blocks) == 1, (
    f"the batch published while the wait timed out was not read: {items}"
  )
  files, segments, _ = blocks[0]
  assert segments.tolist() == [6, 7]
  assert files.tolist() == [0, 0]
  assert worker._ring_flags.tolist() == [WRITABLE_FLAG] * _N_SLOTS
  # The hand-off is passed on for the next worker, as on every way out.
  assert worker._sem_filled.permits == 1


def test_a_wait_that_times_out_with_nothing_left_leaves_at_once() -> None:
  worker = _worker(lambda worker: worker._all_producers_finished.set())
  worker._all_producers_finished.set()

  WorkerBase.run_main(worker)

  assert worker._out_q.items == [None]
  assert worker._sem_filled.timed_waits == 1, "waited again after the run had ended"
  assert worker._ring_flags.tolist() == [WRITABLE_FLAG] * _N_SLOTS


def test_a_cancelled_run_leaves_without_taking_a_batch() -> None:
  def publish_then_cancel(worker: SimpleNamespace) -> None:
    _publish_last_batch_and_finish(worker)
    worker._cancel_event.set()

  worker = _worker(publish_then_cancel)

  WorkerBase.run_main(worker)

  assert worker._out_q.items == [], "a cancelled worker sends nothing"
  assert worker._sem_filled.permits == 2, "a cancelled worker took a permit"
  assert worker._ring_flags[1] == READABLE_FLAG
