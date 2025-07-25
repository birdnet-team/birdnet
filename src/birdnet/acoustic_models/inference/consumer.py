from __future__ import annotations

import multiprocessing as mp
from multiprocessing.synchronize import Event
from queue import Empty

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.tensor import TensorBase


class Consumer:
  def __init__(
    self,
    n_workers: int,
    worker_queue: mp.Queue,
    tensor: TensorBase,
    cancel_event: Event,
  ):
    self._n_workers = n_workers
    self._queue = worker_queue
    self._tensor = tensor
    self._cancel_event = cancel_event
    self._logger = bn_logging.get_logger(__name__)

  def __call__(self):
    finished_workers = 0
    n_received_predictions = 0
    while finished_workers < self._n_workers:
      if self._cancel_event.is_set():
        self._logger.debug("CONSUMER - Cancel event set. Exiting.")
        return

      received_block = None
      while True:
        try:
          received_block = self._queue.get(timeout=1.0)
          break
        except Empty:
          if self._cancel_event.is_set():
            self._logger.debug("CONSUMER - Cancel event set. Exiting.")
            return

      if self._cancel_event.is_set():
        self._logger.debug("CONSUMER - Cancel event set. Exiting.")
        return

      got_stop_signal_from_worker = received_block is None
      if got_stop_signal_from_worker:
        self._logger.debug(
          f"CONSUMER - Received stop signal from worker. Finished workers: {finished_workers + 1}."
        )
        finished_workers += 1
        continue

      assert received_block is not None

      block = received_block
      n_received_predictions += 1
      self._logger.debug(
        f"CONSUMER - Received block with {len(block)} values from worker. Total received: {n_received_predictions}"
      )
      self._tensor.write_block(*block)
    