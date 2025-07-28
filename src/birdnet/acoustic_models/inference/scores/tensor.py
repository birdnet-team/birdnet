from __future__ import annotations

import multiprocessing as mp

import numpy as np
from numpy.typing import DTypeLike

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.helper import uint_dtype_for


class ScoresTensor(TensorBase):
  def __init__(
    self,
    n_files: int,
    top_k: int,
    n_species: int,
    prob_dtype: DTypeLike,
    files_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: mp.RawValue,  # TODO: watch max_n_segments instead
  ) -> None:
    self._logger = bn_logging.get_logger(__name__)

    self._files_dtype = files_dtype
    self._segment_indices_dtype = segment_indices_dtype
    self._top_k = top_k
    self._max_segment_index = max_segment_index

    initial_n_segments = max_segment_index.value + 1

    self._species_ids = np.empty(
      (n_files, initial_n_segments, self._top_k),
      dtype=uint_dtype_for(
        max(0, n_species - 1),
      ),
    )

    self._species_probs = np.empty(
      (n_files, initial_n_segments, self._top_k), dtype=prob_dtype
    )

    self._species_masked = np.full(
      (n_files, initial_n_segments, self._top_k), True, dtype=bool
    )
    self._logger.debug(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  @property
  def memory_usage_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
    ) / 1024**2

  @property
  def current_n_segments(self) -> int:
    return self._species_ids.shape[1]

  def _ensure_capacity(self, needed_n_segments: int) -> None:
    if needed_n_segments <= self.current_n_segments:
      return

    old_n_segments = self.current_n_segments

    self._species_ids.resize(
      (self._species_ids.shape[0], needed_n_segments, self._species_ids.shape[2]),
      refcheck=False,
    )
    self._species_probs.resize(
      (self._species_probs.shape[0], needed_n_segments, self._species_probs.shape[2]),
      refcheck=False,
    )

    self._species_masked.resize(
      (self._species_masked.shape[0], needed_n_segments, self._species_masked.shape[2]),
      refcheck=False,
    )
    # --- Initialisiere NUR den neu angehängten Bereich ----------------
    self._species_masked[:, old_n_segments:needed_n_segments, :] = True

    self._logger.debug(
      f"[resized] from {old_n_segments} to {needed_n_segments} segments. Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    top_k_species: np.ndarray,  # 2dim
    top_k_scores: np.ndarray,  # 2dim
    top_k_mask: np.ndarray,  # 2dim
  ) -> None:
    assert file_indices.dtype == self._files_dtype
    assert top_k_species.dtype == self._species_ids.dtype
    assert top_k_scores.dtype == self._species_probs.dtype
    assert top_k_mask.dtype == self._species_masked.dtype
    assert segment_indices.dtype == self._segment_indices_dtype
    block_max_segment_idx = segment_indices.max()
    max_segment_size = max(block_max_segment_idx, self._max_segment_index.value) + 1
    self._ensure_capacity(max_segment_size)
    self._species_ids[file_indices, segment_indices] = top_k_species
    self._species_probs[file_indices, segment_indices] = top_k_scores
    self._species_masked[file_indices, segment_indices] = top_k_mask

