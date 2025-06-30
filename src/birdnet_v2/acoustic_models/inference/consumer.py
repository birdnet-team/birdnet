from __future__ import annotations

import multiprocessing as mp
from multiprocessing.synchronize import Event
from queue import Empty

import birdnet_v2.logging_utils as bn_logging
from birdnet_v2.acoustic_models.inference.species_tensor import SpeciesTensor


class Consumer:
  def __init__(
    self,
    n_workers: int,
    worker_queue: mp.Queue,
    species_tensor: SpeciesTensor,
    max_chunk_index: mp.RawValue,
    cancel_event: Event,
  ):
    self._n_workers = n_workers
    self._queue = worker_queue
    self._tensor = species_tensor
    self._max_chunk_index = max_chunk_index
    self._cancel_event = cancel_event
    self._logger = bn_logging.get_logger(__name__)

  def __call__(self):
    finished_workers = 0
    n_received_predictions = 0
    while finished_workers < self._n_workers:
      cancel = False
      data = None
      while True:
        try:
          data = self._queue.get(timeout=1.0)
          break
        except Empty:
          if self._cancel_event.is_set():
            cancel = True
            break

      if self._cancel_event.is_set():
        cancel = True

      if cancel:
        self._logger.debug("CONSUMER - Cancel event set. Exiting.")
        break

      assert data is not None

      got_stop_signal_from_worker = data is None
      if got_stop_signal_from_worker:
        finished_workers += 1
        continue

      file_indices, chunk_indices, top_k_species, top_k_scores, top_k_mask = data
      n_received_predictions += top_k_species.shape[0]
      self._logger.debug(
        f"CONSUMER - Received data from worker. Total received: {n_received_predictions}. Chunks: {chunk_indices}"
      )
      self._tensor.write_block(
        file_indices,
        chunk_indices,
        top_k_species,
        top_k_scores,
        top_k_mask,
        self._max_chunk_index.value,
      )
