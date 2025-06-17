from __future__ import annotations

import logging

import numpy as np


class SpeciesTensor:
  def __init__(self, n_files: int, n_chunks: int, top_k: int):
    self._top_k = top_k

    self._species_ids = np.empty((n_files, n_chunks, self._top_k), dtype=np.uint16)
    self._species_probs = np.empty((n_files, n_chunks, self._top_k), dtype=np.float32)
    self._species_masked = np.full((n_files, n_chunks, self._top_k), True, dtype=bool)

    logger = logging.getLogger(__name__)
    logger.debug(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  @property
  def memory_usage_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
    ) / (1024 * 1024)

  @property
  def current_n_chunks(self) -> int:
    return self._species_ids.shape[1]

  def ensure_capacity(self, needed_n_chunks: int):
    if needed_n_chunks <= self.current_n_chunks:
      return

    old_n_chunks = self.current_n_chunks

    self._species_ids.resize(
      (self._species_ids.shape[0], needed_n_chunks, self._species_ids.shape[2]),
      refcheck=False,
    )
    self._species_probs.resize(
      (self._species_probs.shape[0], needed_n_chunks, self._species_probs.shape[2]),
      refcheck=False,
    )

    self._species_masked.resize(
      (self._species_masked.shape[0], needed_n_chunks, self._species_masked.shape[2]),
      refcheck=False,
    )
    # --- Initialisiere NUR den neu angehängten Bereich ----------------
    self._species_masked[:, old_n_chunks:needed_n_chunks, :] = True

    logger = logging.getLogger(__name__)
    logger.debug(
      f"[resized] from {old_n_chunks} → {needed_n_chunks} chunks. Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    chunk_indices: np.ndarray,
    top_k_species: np.ndarray,  # 2dim
    top_k_scores: np.ndarray,  # 2dim
    top_k_mask: np.ndarray,  # 2dim
  ):
    max_chunk_size = max(chunk_indices) + 1
    self.ensure_capacity(max_chunk_size)
    self._species_ids[file_indices, chunk_indices] = top_k_species
    self._species_probs[file_indices, chunk_indices] = top_k_scores
    self._species_masked[file_indices, chunk_indices] = top_k_mask
