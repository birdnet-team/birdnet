from __future__ import annotations

import multiprocessing as mp

import numpy as np
from numpy.typing import DTypeLike

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.tensor import TensorBase


class EmbeddingsTensor(TensorBase):
  def __init__(
    self,
    n_files: int,
    emb_dim: int,
    emb_dtype: DTypeLike,
    files_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: mp.RawValue,
  ) -> None:
    self._logger = bn_logging.get_logger(__name__)

    self._files_dtype = files_dtype
    self._segment_indices_dtype = segment_indices_dtype
    self._max_segment_index = max_segment_index

    initial_n_segments = max_segment_index.value + 1
    self._emb = np.empty((n_files, initial_n_segments, emb_dim), dtype=emb_dtype)
    self._emb_masked = np.full(self._emb.shape, True, dtype=bool)

    self._logger.debug(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  @property
  def memory_usage_mb(self) -> float:
    return (self._emb.nbytes + self._emb_masked.nbytes) / 1024**2

  @property
  def current_n_segments(self) -> int:
    return self._emb.shape[1]

  def _ensure_capacity(self, needed_n_segments: int) -> None:
    if needed_n_segments <= self.current_n_segments:
      return

    old_n_segments = self.current_n_segments

    self._emb.resize(
      (
        self._emb.shape[0],
        needed_n_segments,
        self._emb.shape[2],
      ),
      refcheck=False,
    )

    self._emb_masked.resize(
      (self._emb_masked.shape[0], needed_n_segments, self._emb_masked.shape[2]),
      refcheck=False,
    )
    self._emb_masked[:, old_n_segments:needed_n_segments, :] = True

    self._logger.debug(
      f"[resized] from {old_n_segments} to {needed_n_segments} segments. Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    emb: np.ndarray,  # 2dim
  ) -> None:
    assert file_indices.dtype == self._files_dtype
    assert emb.dtype == self._emb.dtype
    assert segment_indices.dtype == self._segment_indices_dtype
    block_max_segment_idx = segment_indices.max()
    max_segment_size = max(block_max_segment_idx, self._max_segment_index.value) + 1
    self._ensure_capacity(max_segment_size)
    self._emb[file_indices, segment_indices] = emb
    self._emb_masked[file_indices, segment_indices] = False
