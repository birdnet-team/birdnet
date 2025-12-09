from __future__ import annotations

import multiprocessing as mp

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic_models.inference.tensor import AcousticTensorBase
from birdnet.acoustic_models.inference_pipeline.logs import get_logger_from_session
from birdnet.helper import get_uint_dtype


class AcousticPredictionTensor(AcousticTensorBase):
  def __init__(
    self,
    session_id: str,
    n_inputs: int,
    top_k: int,
    n_species: int,
    half_precision: bool,
    input_indices_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: mp.RawValue,  # TODO: watch max_n_segments instead # type: ignore
  ) -> None:
    self._session_id = session_id
    self._logger = get_logger_from_session(session_id, __name__)

    self._input_indices_dtype = input_indices_dtype
    self._segment_indices_dtype = segment_indices_dtype
    self._top_k = top_k
    self._max_segment_index = max_segment_index

    initial_n_segments = max_segment_index.value

    self._species_ids = np.empty(
      (n_inputs, initial_n_segments, self._top_k),
      dtype=get_uint_dtype(
        max(0, n_species - 1),
      ),
    )

    _species_probs_type = np.float16 if half_precision else np.float32

    self._species_probs = np.empty(
      (n_inputs, initial_n_segments, self._top_k), dtype=_species_probs_type
    )

    self._species_masked = np.full(
      (n_inputs, initial_n_segments, self._top_k), True, dtype=bool
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

    # --- Initialize ONLY the newly appended area ----------------
    self._species_masked[:, old_n_segments:needed_n_segments, :] = True

    self._logger.debug(
      f"[resized] from {old_n_segments} to {needed_n_segments} segments. Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def write_block(
    self,
    input_indices: np.ndarray,
    segment_indices: np.ndarray,
    top_k_species: np.ndarray,  # 2dim
    top_k_scores: np.ndarray,  # 2dim
    top_k_mask: np.ndarray,  # 2dim
  ) -> None:
    assert input_indices.dtype == self._input_indices_dtype
    assert top_k_species.dtype == self._species_ids.dtype
    assert top_k_scores.dtype == self._species_probs.dtype
    assert top_k_mask.dtype == self._species_masked.dtype
    assert segment_indices.dtype == self._segment_indices_dtype
    block_max_segment_idx = segment_indices.max()
    max_segment_size = max(block_max_segment_idx, self._max_segment_index.value) + 1
    self._ensure_capacity(max_segment_size)
    self._species_ids[input_indices, segment_indices] = top_k_species
    self._species_probs[input_indices, segment_indices] = top_k_scores
    self._species_masked[input_indices, segment_indices] = top_k_mask

  def set_unprocessable_inputs(self, unprocessable_inputs: set[int]) -> None:
    super().set_unprocessable_inputs(unprocessable_inputs)
    self._species_probs[self._unprocessable_inputs, :, :] = 0.0
    self._species_ids[self._unprocessable_inputs, :, :] = 0
    self._species_masked[self._unprocessable_inputs, :, :] = True
