from __future__ import annotations

import queue
import sys

import numpy as np

from birdnet_v3.producer import Producer
from birdnet_v3.worker import EMPTY_ID, Worker


class SpeciesTensor:
  def __init__(self, n_files: int, n_chunks: int, top_k: int):
    self._top_k = top_k
    self._alloc(n_files, n_chunks)

  def _alloc(self, F: int, max_chunk_size: int):
    self._species_ids = np.full(
      (F, max_chunk_size, self._top_k), EMPTY_ID, dtype=np.uint16
    )
    self._species_probs = np.zeros((F, max_chunk_size, self._top_k), dtype=np.float16)
    self._prediction_mask = np.ones((F, max_chunk_size, self._top_k), dtype=bool)

  @property
  def current_n_chunks(self) -> int:
    return self._species_ids.shape[1]

  def ensure_capacity(self, needed_n_chunks: int):
    if needed_n_chunks <= self.current_n_chunks:
      return

    print(f"[resize] → {needed_n_chunks} chunks", file=sys.stderr)
    self._species_ids.resize(
      (self._species_ids.shape[0], needed_n_chunks, self._species_ids.shape[2]),
      refcheck=False,
    )
    self._species_probs.resize(
      (self._species_probs.shape[0], needed_n_chunks, self._species_probs.shape[2]),
      refcheck=False,
    )
    self._prediction_mask.resize(
      (self._prediction_mask.shape[0], needed_n_chunks, self._prediction_mask.shape[2]),
      refcheck=False,
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    chunk_indices: np.ndarray,
    species_indicies: np.ndarray,  # 2dim
    species_probs: np.ndarray,  # 2dim
    pred_msk: np.ndarray,  # 2dim
  ):
    max_chunk_size = max(chunk_indices) + 1
    self.ensure_capacity(max_chunk_size)
    self._species_ids[file_indices, chunk_indices] = species_indicies
    self._species_probs[file_indices, chunk_indices] = species_probs
    self._prediction_mask[file_indices, chunk_indices] = pred_msk

  def get_at(self, file_index, chunk_index):
    valid = ~self._prediction_mask[file_index, chunk_index]
    ids = self._species_ids[file_index, chunk_index][valid]
    probs = self._species_probs[file_index, chunk_index][valid]
    return [(i, float(p)) for i, p in zip(ids, probs)]


class Consumer:
  def __init__(self, producer: Producer, worker: Worker, init_w: int = 512):
    self._producer = producer
    self._worker = worker
    self.init_w = init_w

  def consume(self):
    n_files = len(self._producer._files)
    tensor = SpeciesTensor(n_files, self.init_w, self._worker.top_k)
    live = len(self._worker.workers)
    while live > 0:
      try:
        file_indices, chunk_indices, species_indicies, species_probs, pred_msk = (
          self._worker.queue.get_nowait()
        )
      except queue.Empty:
        # check alive
        live = sum(p.is_alive() for p in self._worker.workers)
        continue
      tensor.write_block(
        file_indices, chunk_indices, species_indicies, species_probs, pred_msk
      )
    return tensor
