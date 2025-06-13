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
from birdnet_v2.inference.worker import EMPTY_ID, EMPTY_PRED, ChildWorker, Worker


class SpeciesTensor:
  def __init__(self, n_files: int, n_chunks: int, top_k: int):
    self._top_k = top_k

    self._species_ids = np.full(
      (n_files, n_chunks, self._top_k), EMPTY_ID, dtype=np.uint16
    )
    self._species_probs = np.full(
      (n_files, n_chunks, self._top_k), EMPTY_PRED, dtype=np.float32
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  @property
  def memory_usage_mb(self) -> float:
    return (self._species_ids.nbytes + self._species_probs.nbytes) / (1024 * 1024)

  @property
  def current_n_chunks(self) -> int:
    return self._species_ids.shape[1]

  def ensure_capacity(self, needed_n_chunks: int):
    if needed_n_chunks <= self.current_n_chunks:
      return

    self._species_ids.resize(
      (self._species_ids.shape[0], needed_n_chunks, self._species_ids.shape[2]),
      refcheck=False,
    )
    self._species_probs.resize(
      (self._species_probs.shape[0], needed_n_chunks, self._species_probs.shape[2]),
      refcheck=False,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"[resize] from {self.current_n_chunks} → {needed_n_chunks} chunks")
    logger.info(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  def write_block(
    self,
    file_indices: np.ndarray,
    chunk_indices: np.ndarray,
    species_indicies: np.ndarray,  # 2dim
    species_probs: np.ndarray,  # 2dim
  ):
    max_chunk_size = max(chunk_indices) + 1
    self.ensure_capacity(max_chunk_size)
    self._species_ids[file_indices, chunk_indices] = species_indicies
    self._species_probs[file_indices, chunk_indices] = species_probs

  # def get_at(self, file_index, chunk_index):
  #   valid = ~self._prediction_mask[file_index, chunk_index]
  #   ids = self._species_ids[file_index, chunk_index][valid]
  #   probs = self._species_probs[file_index, chunk_index][valid]
  #   return [(i, float(p)) for i, p in zip(ids, probs)]
