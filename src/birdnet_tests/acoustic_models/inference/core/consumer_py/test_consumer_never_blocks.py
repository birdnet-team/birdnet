"""The consumer's loop must stay bounded when its queue stops being readable.

The consumer is the run: while it is blocked, no liveness check executes and
teardown is never reached. Before this change it read the results queue with
``Queue.get(timeout=1.0)`` -- whose timeout does not bound the receive -- so a
message a killed worker left half-written stopped the whole session with
nothing raised (issue #83). Reading through ``ThreadedQueueReader`` confines
that block to a sacrificial thread; this test pins the property at the
consumer level, deterministically and on every platform, by wedging the queue
with a stand-in rather than a real torn frame (the real one is covered in
test_threaded_queue_reader.py).
"""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Empty

import numpy as np
import pytest

from birdnet.acoustic.inference.core.consumer import Consumer
from birdnet.acoustic.inference.core.sync import ThreadedQueueReader
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase

pytestmark = pytest.mark.no_tf

_DEADLINE_S = 30.0


class _Tensor(AcousticTensorBase):
  def __init__(self) -> None:
    super().__init__()
    self.blocks: list[tuple] = []

  @property
  def memory_usage_mb(self) -> float:
    return 0.0

  def write_block(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
    self.blocks.append(args)


class _WedgingQueue:
  """Delivers a few blocks, then blocks forever -- a torn frame's shape."""

  def __init__(self, blocks: list[object], release: threading.Event) -> None:
    self._items = list(blocks)
    self._release = release

  def get(self, timeout: float | None = None) -> object:
    if self._items:
      return self._items.pop(0)
    self._release.wait(120.0)
    raise Empty


def _block() -> tuple:
  return (np.zeros(1, np.uint16), np.zeros(1, np.uint32), np.zeros((1, 4)))


def test_a_dead_worker_mid_message_fails_the_run_instead_of_freezing_it() -> None:
  release = threading.Event()
  cancel = threading.Event()
  # Two workers promised; one block arrives, then the queue never yields
  # again -- as after a worker died half-way through writing its next block.
  reader = ThreadedQueueReader(
    _WedgingQueue([_block()], release),  # type: ignore[arg-type]
    "wedging-results",
  )
  tensor = _Tensor()

  def liveness() -> None:
    # Stands in for raise_if_child_died finding the dead worker: it marks the
    # run cancelled and raises, exactly like the real check.
    cancel.set()
    raise ChildProcessError("Worker-0 exited unexpectedly")

  consumer = Consumer(
    session_id="consumer-block-test",
    n_workers=2,
    results=reader,
    tensor=tensor,
    cancel_event=cancel,  # type: ignore[arg-type]
    check_children_alive=liveness,
  )

  finished = threading.Event()

  def run() -> None:
    try:
      consumer()
    finally:
      finished.set()

  threading.Thread(target=run, name="consumer-under-test", daemon=True).start()
  try:
    assert finished.wait(timeout=_DEADLINE_S), (
      f"the consumer did not return within {_DEADLINE_S:.0f} s of its queue "
      f"going unreadable; with a direct Queue.get this is exactly issue #83"
    )
  finally:
    release.set()
    reader.close()

  assert cancel.is_set(), "the liveness check was never consulted"
  assert len(tensor.blocks) == 1, "the block that did arrive must be kept"


def test_a_sentinel_lost_after_the_finish_signal_still_ends_the_run(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The liveness check's blind spot: dead *after* signalling, sentinel unsent.

  `raise_if_child_died` deliberately skips a child whose finish signal is set,
  so a worker killed between signalling and its feeder flushing the sentinel
  is invisible to it -- the case that keeps the old loop waiting forever. Once
  every worker reports finished, a quiet deadline is the only honest wait.
  """
  from birdnet.acoustic.inference.core import consumer as consumer_module

  monkeypatch.setattr(consumer_module, "PROMISED_MESSAGE_DEADLINE_S", 2.0)

  release = threading.Event()
  cancel = threading.Event()
  # One block arrives, one sentinel arrives -- but the second worker's sentinel
  # never does, and the liveness check sees nothing wrong.
  reader = ThreadedQueueReader(
    _WedgingQueue([_block(), None], release),  # type: ignore[arg-type]
    "sentinel-lost",
  )

  consumer = Consumer(
    session_id="sentinel-loss-test",
    n_workers=2,
    results=reader,
    tensor=_Tensor(),
    cancel_event=cancel,  # type: ignore[arg-type]
    check_children_alive=lambda: None,  # everyone looks fine
    all_workers_finished=lambda: True,  # ...and everyone has signalled
  )

  finished = threading.Event()

  def run() -> None:
    try:
      consumer()
    finally:
      finished.set()

  threading.Thread(target=run, name="sentinel-loss-run", daemon=True).start()
  try:
    assert finished.wait(timeout=_DEADLINE_S), (
      "the consumer waited past the quiet deadline for a sentinel that a "
      "signalled-and-killed worker can never send"
    )
  finally:
    release.set()
    reader.close()

  assert cancel.is_set(), "a lost sentinel must fail the run, not end it cleanly"


def test_a_marker_lost_after_the_finish_signal_still_ends_the_run(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The same blind spot on the completion-marker path.

  All segments are written and every worker is done, but one producer's
  completion marker was in its feeder when the producer was killed -- after
  its finish signal, so the liveness check passes it over. Finalization must
  give up at the deadline and fail the run rather than wait for a marker that
  cannot come (and rather than fire the per-file callback with made-up data).
  """
  import queue as pyqueue

  from birdnet.acoustic.inference.core import consumer as consumer_module

  monkeypatch.setattr(consumer_module, "PROMISED_MESSAGE_DEADLINE_S", 2.0)

  release = threading.Event()
  cancel = threading.Event()
  results = ThreadedQueueReader(
    _WedgingQueue([], release),
    "finalize-results",  # type: ignore[arg-type]
  )
  markers = ThreadedQueueReader(
    _WedgingQueue([], release),
    "finalize-markers",  # type: ignore[arg-type]
  )
  dispatch: pyqueue.Queue = pyqueue.Queue()

  consumer = Consumer(
    session_id="marker-loss-test",
    n_workers=0,  # the main loop is not what this test is about
    results=results,
    tensor=_Tensor(),
    cancel_event=cancel,  # type: ignore[arg-type]
    n_inputs=1,
    inputs=[Path("one-file.wav")],
    markers=markers,
    completion_dispatch_queue=dispatch,
    check_children_alive=lambda: None,  # the dead producer looks finished
  )

  finished = threading.Event()

  def run() -> None:
    try:
      consumer()
    finally:
      finished.set()

  threading.Thread(target=run, name="marker-loss-run", daemon=True).start()
  try:
    assert finished.wait(timeout=_DEADLINE_S), (
      "marker finalization waited past its deadline for a marker a "
      "signalled-and-killed producer can never send"
    )
  finally:
    release.set()
    results.close()
    markers.close()

  assert cancel.is_set(), "a lost marker must fail the run"
  assert dispatch.get_nowait() is None, (
    "the dispatcher must still receive its end-of-run sentinel, or its join "
    "waits forever"
  )
