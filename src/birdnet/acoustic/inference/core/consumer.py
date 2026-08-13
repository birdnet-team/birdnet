from __future__ import annotations

import queue
from collections.abc import Callable
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty

import numpy as np

from birdnet.acoustic.inference.core.logs import get_logger_from_session
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase


class Consumer:
  def __init__(
    self,
    session_id: str,
    n_workers: int,
    worker_queue: Queue,
    tensor: AcousticTensorBase,
    cancel_event: Event,
    *,
    n_inputs: int = 0,
    inputs: list[Path] | None = None,
    completion_marker_queue: Queue | None = None,
    completion_dispatch_queue: queue.Queue | None = None,
    check_children_alive: Callable[[], None] | None = None,
  ) -> None:
    self._n_workers = n_workers
    self._queue = worker_queue
    self._tensor = tensor
    self._cancel_event = cancel_event
    self._logger = get_logger_from_session(session_id, __name__)
    # Called on every idle poll below; raises if a worker died without putting
    # its sentinel on the queue, which would otherwise wait here forever.
    self._check_children_alive = check_children_alive

    # Per-file completion tracking (``on_file_complete``). Fully inert unless a
    # marker queue is provided, so the default hot path is byte-for-byte
    # unchanged.
    self._track_completion = completion_marker_queue is not None
    self._marker_queue = completion_marker_queue
    self._dispatch_queue = completion_dispatch_queue
    self._inputs = inputs
    self._n_inputs = n_inputs
    if self._track_completion:
      assert self._dispatch_queue is not None
      assert self._inputs is not None
      assert self._n_inputs == len(self._inputs)
      self._written = np.zeros(n_inputs, dtype=np.int64)
      self._expected = np.zeros(n_inputs, dtype=np.int64)
      self._duration = np.zeros(n_inputs, dtype=np.float64)
      self._invalid = np.zeros(n_inputs, dtype=bool)
      self._marker_received = np.zeros(n_inputs, dtype=bool)
      self._n_markers = 0
      self._pending: set[int] = set()

  def _log(self, message: str) -> None:
    self._logger.debug(f"C: {message}")

  def __call__(self) -> None:
    try:
      self._run_main_loop()
    except Exception as e:
      self._logger.exception(
        "Consumer encountered an exception.", exc_info=e, stack_info=True
      )
      self._cancel_event.set()
    finally:
      if self._track_completion:
        self._end_dispatch()

  def _run_main_loop(self) -> None:
    finished_workers = 0
    n_received_predictions = 0
    while finished_workers < self._n_workers:
      if self._cancel_event.is_set():
        self._log("Cancel event set. Exiting.")
        return

      received_block = None
      while True:
        try:
          received_block = self._queue.get(timeout=1.0)
          break
        except Empty:
          if self._cancel_event.is_set():
            self._log("Cancel event set. Exiting.")
            return
          if self._check_children_alive is not None:
            self._check_children_alive()

      if self._cancel_event.is_set():
        self._log("Cancel event set. Exiting.")
        return

      got_stop_signal_from_worker = received_block is None
      if got_stop_signal_from_worker:
        self._log(
          f"Received stop signal from worker. Finished workers: {finished_workers + 1}."
        )
        finished_workers += 1
        continue

      assert received_block is not None

      block = received_block
      n_received_predictions += 1
      self._log(
        f"Received block with {len(block)} values from worker. "
        f"Total received: {n_received_predictions}"
      )
      self._tensor.write_block(*block)

      if self._track_completion:
        # block[0] holds the per-segment input indices of this batch. bincount
        # is C-speed (unlike np.add.at) so this stays cheap on the hot path.
        self._written += np.bincount(block[0], minlength=self._n_inputs)
        self._drain_markers()
        self._dispatch_ready()

  # -- per-file completion helpers (only used when tracking is enabled) --------

  def _apply_marker(self, marker: tuple[int, int, bool, float]) -> None:
    idx, n_emitted, is_invalid, duration = marker
    if self._marker_received[idx]:
      return
    self._marker_received[idx] = True
    self._expected[idx] = n_emitted
    self._invalid[idx] = is_invalid
    self._duration[idx] = duration
    self._n_markers += 1
    self._pending.add(idx)

  def _drain_markers(self) -> None:
    assert self._marker_queue is not None
    while True:
      try:
        marker = self._marker_queue.get_nowait()
      except Empty:
        break
      self._apply_marker(marker)

  def _finalize_markers(self) -> None:
    # After all workers finished, every segment has been written; block until we
    # have every producer marker so no completed file is missed.
    assert self._marker_queue is not None
    while self._n_markers < self._n_inputs:
      if self._cancel_event.is_set():
        return
      try:
        marker = self._marker_queue.get(timeout=1.0)
      except Empty:
        continue
      self._apply_marker(marker)

  def _dispatch_ready(self) -> None:
    if not self._pending:
      return
    done: list[int] = []
    for idx in self._pending:
      if self._invalid[idx]:
        # Invalid/partial files are reported as zero-detection, matching how the
        # aggregate result masks unprocessable inputs.
        self._emit(idx, valid=False)
        done.append(idx)
      elif self._written[idx] >= self._expected[idx]:
        self._emit(idx, valid=True)
        done.append(idx)
    for idx in done:
      self._pending.discard(idx)

  def _emit(self, idx: int, *, valid: bool) -> None:
    assert self._dispatch_queue is not None
    assert self._inputs is not None
    n_segments = int(self._expected[idx]) if valid else 0
    # Runs on this (consumer) thread, so the tensor is never read and written
    # concurrently; the copied arrays are what crosses to the dispatcher thread.
    # The array tuple is tensor-specific (predictions vs embeddings); the
    # strategy that built the tensor knows how to turn it back into a result.
    arrays = self._tensor.copy_file_slice(idx, n_segments)  # type: ignore[attr-defined]
    self._dispatch_queue.put(
      (self._inputs[idx], arrays, not valid, float(self._duration[idx]))
    )

  def _end_dispatch(self) -> None:
    if not self._cancel_event.is_set():
      self._finalize_markers()
      self._dispatch_ready()
    assert self._dispatch_queue is not None
    # Sentinel tells the dispatcher this run is done.
    self._dispatch_queue.put(None)
