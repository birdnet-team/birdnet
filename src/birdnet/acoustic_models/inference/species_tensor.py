from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

import birdnet.logging_utils as bn_logging
from birdnet.helper import uint_dtype_for


class SpeciesTensor:
  def __init__(
    self,
    n_files: int,
    n_chunks: int,
    top_k: int,
    n_species: int,
    prob_dtype: DTypeLike,
    files_dtype: DTypeLike,
    chunk_indices_dtype: DTypeLike,
  ) -> None:
    self._logger = bn_logging.get_logger(__name__)

    self._files_dtype = files_dtype
    self._chunk_indices_dtype = chunk_indices_dtype
    self._top_k = top_k

    self._species_ids = np.empty(
      (n_files, n_chunks, self._top_k),
      dtype=uint_dtype_for(
        max(0, n_species - 1),
      ),
    )

    self._species_probs = np.empty((n_files, n_chunks, self._top_k), dtype=prob_dtype)

    self._species_masked = np.full((n_files, n_chunks, self._top_k), True, dtype=bool)
    self._logger.debug(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  @property
  def memory_usage_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
    ) / 1024**2

  @property
  def current_n_chunks(self) -> int:
    return self._species_ids.shape[1]

  def _ensure_capacity(self, needed_n_chunks: int) -> None:
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

    self._logger.debug(
      f"[resized] from {old_n_chunks} to {needed_n_chunks} chunks. Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    chunk_indices: np.ndarray,
    top_k_species: np.ndarray,  # 2dim
    top_k_scores: np.ndarray,  # 2dim
    top_k_mask: np.ndarray,  # 2dim
    global_max_chunk_idx: int,
  ) -> None:
    assert file_indices.dtype == self._files_dtype
    assert top_k_species.dtype == self._species_ids.dtype
    assert top_k_scores.dtype == self._species_probs.dtype
    assert top_k_mask.dtype == self._species_masked.dtype
    assert chunk_indices.dtype == self._chunk_indices_dtype
    block_max_chunk_idx = chunk_indices.max()
    max_chunk_size = max(block_max_chunk_idx, global_max_chunk_idx) + 1
    self._ensure_capacity(max_chunk_size)
    self._species_ids[file_indices, chunk_indices] = top_k_species
    self._species_probs[file_indices, chunk_indices] = top_k_scores
    self._species_masked[file_indices, chunk_indices] = top_k_mask
