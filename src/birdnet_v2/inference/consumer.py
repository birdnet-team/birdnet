from __future__ import annotations

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
from birdnet_v2.inference.worker import EMPTY_ID, ChildWorker, Worker


class Consumer:
  def __init__(
    self,
    n_jobs: int,
    worker_queue: mp.Queue,
    species_tensor: SpeciesTensor,
  ):
    self._n_jobs = n_jobs
    self._queue = worker_queue
    self._tensor = species_tensor

  def __call__(self):
    finished_workers = 0
    while finished_workers < self._n_jobs:
      data = self._queue.get()
      was_stop_signal_from_worker = data is None
      if was_stop_signal_from_worker:
        finished_workers += 1
      else:
        file_indices, chunk_indices, species_indicies, species_probs = data
        self._tensor.write_block(
          file_indices, chunk_indices, species_indicies, species_probs
        )
