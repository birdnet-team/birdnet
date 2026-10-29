"""Children and resources are annotated with the protocols, not the raw types.

The protocols only bite where they are used: a child constructor annotated
with a raw ``Queue`` or ``Event`` again would let a raw read or wait through.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from birdnet.acoustic.inference.core.encoding.encoding_worker import EncodingWorker
from birdnet.acoustic.inference.core.perf_tracker import (
  PerformanceTracker,
  ProgressDispatcher,
)
from birdnet.acoustic.inference.core.prediction.prediction_worker import (
  PredictionWorker,
)
from birdnet.acoustic.inference.core.producer import Producer
from birdnet.acoustic.inference.core.sync import ChildSideEvent, ChildToParentQueue
from birdnet.acoustic.inference.core.worker import WorkerBase
from birdnet.acoustic.inference.resources import (
  FileCompletionResources,
  ProducerResources,
  StatisticsResources,
  WorkerResources,
)

pytestmark = pytest.mark.no_tf


class _Hints:
  """One annotation at a time: get_type_hints trips over Synchronized[int]."""

  def __init__(self, owner: type, of_init: bool = True) -> None:
    target = owner.__init__ if of_init else owner
    self._annotations = target.__annotations__
    self._namespace = vars(sys.modules[owner.__module__])

  def __getitem__(self, name: str) -> Any:  # noqa: ANN401 - a type object
    ann = self._annotations[name]
    return eval(ann, self._namespace) if isinstance(ann, str) else ann  # noqa: S307


def _hints(owner: type, of_init: bool = True) -> _Hints:
  return _Hints(owner, of_init)


@pytest.mark.parametrize("worker", [WorkerBase, PredictionWorker, EncodingWorker])
def test_workers(worker: type) -> None:
  hints = _hints(worker)
  assert hints["out_q"] is ChildToParentQueue
  assert hints["wkr_stats_queue"] == ChildToParentQueue | None
  for name in ("cancel_event", "all_producers_finished", "end_event", "finish_signal"):
    assert hints[name] is ChildSideEvent, name


def test_producer() -> None:
  hints = _hints(Producer)
  assert hints["unprocessed_inputs_queue"] is ChildToParentQueue
  assert hints["prod_stats_queue"] == ChildToParentQueue | None
  assert hints["completion_queue"] == ChildToParentQueue | None
  for name in ("cancel_event", "all_finished", "end_event", "finish_signal"):
    assert hints[name] is ChildSideEvent, name


def test_performance_tracker_and_dispatcher() -> None:
  hints = _hints(PerformanceTracker)
  assert hints["perf_res"] is ChildToParentQueue
  assert hints["callback_queue"] == ChildToParentQueue | None
  for name in ("cancel_event", "end_event", "finish_signal"):
    assert hints[name] is ChildSideEvent, name
  assert _hints(ProgressDispatcher)["callback_queue"] is ChildToParentQueue


def test_resources() -> None:
  workers = _hints(WorkerResources, of_init=False)
  assert workers["results_queue"] is ChildToParentQueue
  producers = _hints(ProducerResources, of_init=False)
  assert producers["unprocessed_inputs_queue"] is ChildToParentQueue
  stats = _hints(StatisticsResources, of_init=False)
  assert stats["perf_res_queue"] == ChildToParentQueue | None
  assert stats["callback_queue"] == ChildToParentQueue | None
  completion = _hints(FileCompletionResources, of_init=False)
  assert completion["marker_queue"] == ChildToParentQueue | None
