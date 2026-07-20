from __future__ import annotations

import multiprocessing as mp

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic.inference.core.logs import get_logger_from_session
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase


class AcousticEncodingTensor(AcousticTensorBase):
  def __init__(
    self,
    session_id: str,
    n_inputs: int,
    emb_dim: int,
    half_precision: bool,
    input_indices_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: mp.RawValue,  # type: ignore
  ) -> None:
    self._session_id = session_id
    self._logger = get_logger_from_session(session_id, __name__)

    self._input_indices_dtype = input_indices_dtype
    self._segment_indices_dtype = segment_indices_dtype
    self._max_segment_index = max_segment_index

    initial_n_segments = 0
    if max_segment_index.value > 0:
      initial_n_segments = max_segment_index.value + 1

    emb_dtype = np.float16 if half_precision else np.float32
    self._emb = np.empty((n_inputs, initial_n_segments, emb_dim), dtype=emb_dtype)
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

    new_shape = (
      self._emb.shape[0],
      needed_n_segments,
      self._emb.shape[2],
    )
    new_emb = np.empty(new_shape, dtype=self._emb.dtype)
    new_emb_masked = np.full(new_shape, True, dtype=self._emb_masked.dtype)

    new_emb[:, :old_n_segments, :] = self._emb
    new_emb_masked[:, :old_n_segments, :] = self._emb_masked

    self._emb = new_emb
    self._emb_masked = new_emb_masked

    self._logger.debug(
      f"[resized] from {old_n_segments} to {needed_n_segments} segments. "
      f"Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    emb: np.ndarray,  # 2dim
  ) -> None:
    assert file_indices.dtype == self._input_indices_dtype
    assert emb.dtype == self._emb.dtype
    assert segment_indices.dtype == self._segment_indices_dtype
    block_max_segment_idx = segment_indices.max()
    max_segment_size = max(block_max_segment_idx, self._max_segment_index.value) + 1
    self._ensure_capacity(max_segment_size)
    self._emb[file_indices, segment_indices] = emb
    self._emb_masked[file_indices, segment_indices] = False

  def set_unprocessable_inputs(self, unprocessable_inputs: set[int]) -> None:
    super().set_unprocessable_inputs(unprocessable_inputs)
    self._emb[self._unprocessable_inputs, :, :] = 0
    self._emb_masked[self._unprocessable_inputs, :, :] = True

  def copy_file_slice(
    self, file_idx: int, n_segments: int
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return an independent copy of a single file's embedding rows.

    The returned arrays have shape ``(1, n_segments, emb_dim)`` so they can back
    a single-file result. Copying (rather than viewing) makes the data safe to
    hand to another thread while this tensor keeps being written/resized.

    Must be called from the same thread that writes the tensor (the consumer);
    ``n_segments`` must not exceed the segments already written for the file.
    """
    assert 0 <= n_segments <= self.current_n_segments
    emb = self._emb[file_idx, :n_segments].copy()[np.newaxis]
    emb_masked = self._emb_masked[file_idx, :n_segments].copy()[np.newaxis]
    return emb, emb_masked


class PrebuiltEncodingTensor(AcousticTensorBase):
  """Minimal tensor holder wrapping already-materialised per-file embeddings.

  Used to build a single-file ``AcousticFileEncodingResult`` from the slice
  copied out of the shared result tensor, without re-running any inference.
  """

  def __init__(self, emb: np.ndarray, emb_masked: np.ndarray) -> None:
    super().__init__()
    self._emb = emb
    self._emb_masked = emb_masked

  @property
  def memory_usage_mb(self) -> float:
    return (self._emb.nbytes + self._emb_masked.nbytes) / 1024**2

  def write_block(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
    raise NotImplementedError("PrebuiltEncodingTensor is read-only.")
