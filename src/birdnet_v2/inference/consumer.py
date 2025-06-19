from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from birdnet_v2.inference.producer import Producer
from birdnet_v2.inference.species_tensor import SpeciesTensor


class Consumer:
  def __init__(
    self,
    n_jobs: int,
    worker_queue: mp.Queue,
    species_tensor: SpeciesTensor,
    max_chunk_index: mp.RawValue,
  ):
    self._n_jobs = n_jobs
    self._queue = worker_queue
    self._tensor = species_tensor
    self._max_chunk_index = max_chunk_index

  def __call__(self):
    finished_workers = 0
    received_predictions = 0
    logger = logging.getLogger(__name__)
    while finished_workers < self._n_jobs:
      data = self._queue.get()
      was_stop_signal_from_worker = data is None
      if was_stop_signal_from_worker:
        finished_workers += 1
      else:
        file_indices, chunk_indices, top_k_species, top_k_scores, top_k_mask = data
        received_predictions += top_k_species.shape[0]
        logger.debug(
          f"CONSUMER - Received data from worker. Total received: {received_predictions}. Chunks: {chunk_indices}"
        )
        self._tensor.write_block(
          file_indices,
          chunk_indices,
          top_k_species,
          top_k_scores,
          top_k_mask,
          self._max_chunk_index.value,
        )
