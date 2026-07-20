from __future__ import annotations

import multiprocessing as mp

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic.inference.core.logs import get_logger_from_session
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase
from birdnet.utils.helper import get_uint_dtype


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

    initial_n_segments = 0
    if max_segment_index.value > 0:
      initial_n_segments = max_segment_index.value + 1

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

    new_shape = (
      self._species_ids.shape[0],
      needed_n_segments,
      self._species_ids.shape[2],
    )
    new_species_ids = np.empty(new_shape, dtype=self._species_ids.dtype)
    new_species_probs = np.empty(new_shape, dtype=self._species_probs.dtype)
    new_species_masked = np.full(new_shape, True, dtype=self._species_masked.dtype)

    new_species_ids[:, :old_n_segments, :] = self._species_ids
    new_species_probs[:, :old_n_segments, :] = self._species_probs
    new_species_masked[:, :old_n_segments, :] = self._species_masked

    self._species_ids = new_species_ids
    self._species_probs = new_species_probs
    self._species_masked = new_species_masked

    self._logger.debug(
      f"[resized] from {old_n_segments} to {needed_n_segments} segments. "
      f"Resulting array allocated: {self.memory_usage_mb:.2f} MB"
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

  def copy_file_slice(
    self, file_idx: int, n_segments: int
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an independent copy of a single file's tensor rows.

    The returned arrays have shape ``(1, n_segments, top_k)`` so they can back a
    single-file result. Copying (rather than viewing) makes the data safe to
    hand to another thread while this tensor keeps being written/resized.

    Must be called from the same thread that writes the tensor (the consumer);
    ``n_segments`` must not exceed the segments already written for the file.
    """
    assert 0 <= n_segments <= self.current_n_segments
    ids = self._species_ids[file_idx, :n_segments].copy()[np.newaxis]
    probs = self._species_probs[file_idx, :n_segments].copy()[np.newaxis]
    masked = self._species_masked[file_idx, :n_segments].copy()[np.newaxis]
    return ids, probs, masked


class PrebuiltPredictionTensor(AcousticTensorBase):
  """Minimal tensor holder wrapping already-materialised per-file arrays.

  Used to build a single-file ``AcousticFilePredictionResult`` from the slice
  copied out of the shared result tensor, without re-running any inference.
  """

  def __init__(
    self,
    species_ids: np.ndarray,
    species_probs: np.ndarray,
    species_masked: np.ndarray,
  ) -> None:
    super().__init__()
    self._species_ids = species_ids
    self._species_probs = species_probs
    self._species_masked = species_masked

  @property
  def memory_usage_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
    ) / 1024**2

  def write_block(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
    raise NotImplementedError("PrebuiltPredictionTensor is read-only.")
