from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic.inference.core.logs import get_logger_from_session
from birdnet.utils.dtypes import UIntCType, get_uint_dtype


class AcousticTensorBase:
  def __init__(self) -> None:
    self._unprocessable_inputs: np.ndarray | None = None

  @property
  @abstractmethod
  def memory_usage_mb(self) -> float: ...

  @abstractmethod
  def write_block(self, *args, **kwargs) -> None: ...

  @abstractmethod
  def copy_file_slice(self, file_idx: int, n_segments: int) -> tuple[np.ndarray, ...]:
    """Return an independent copy of a single file's rows of every array.

    The returned arrays have shape ``(1, n_segments, width)`` so they can back a
    single-file result. Copying (rather than viewing) makes the data safe to
    hand to another thread while this tensor keeps being written/resized.

    Must be called from the same thread that writes the tensor (the consumer);
    ``n_segments`` must not exceed the segments already written for the file.
    """

  def set_unprocessable_inputs(self, unprocessable_inputs: set[int]) -> None:
    if len(unprocessable_inputs) == 0:
      unprocessable_inputs_np = np.array([], dtype=np.uint8)
    else:
      unprocessable_inputs_np = np.array(
        sorted(unprocessable_inputs), dtype=get_uint_dtype(max(unprocessable_inputs))
      )
    self._unprocessable_inputs = unprocessable_inputs_np

  @property
  def unprocessable_inputs(self) -> np.ndarray:
    assert self._unprocessable_inputs is not None
    return self._unprocessable_inputs


def _allocate(shape: tuple[int, int, int], dtype: DTypeLike) -> np.ndarray:
  # Masks start fully masked; value arrays are only read where unmasked.
  if np.dtype(dtype) == np.bool_:
    return np.full(shape, True, dtype=bool)
  return np.empty(shape, dtype=dtype)


class AcousticArrayTensorBase(AcousticTensorBase):
  """Named result arrays of shape ``(n_inputs, n_segments, width)``.

  The arrays grow along the segment axis as blocks arrive. ``_ARRAYS`` lists
  them as ``(attribute name, value written into an unprocessable input's rows)``
  in the order ``arrays`` and ``copy_file_slice`` return them; the constructor
  takes their ``(attribute name, width, dtype)`` in the same order.
  """

  _ARRAYS: ClassVar[tuple[tuple[str, int | float | bool], ...]]

  def __init__(
    self,
    session_id: str,
    n_inputs: int,
    input_indices_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: UIntCType,
    array_specs: Sequence[tuple[str, int, DTypeLike]],
  ) -> None:
    super().__init__()
    assert [name for name, _, _ in array_specs] == [name for name, _ in self._ARRAYS]
    self._session_id = session_id
    self._logger = get_logger_from_session(session_id, type(self).__module__)

    self._input_indices_dtype = input_indices_dtype
    self._segment_indices_dtype = segment_indices_dtype
    self._max_segment_index = max_segment_index

    initial_n_segments = 0
    if max_segment_index.value > 0:
      initial_n_segments = max_segment_index.value + 1

    for name, width, dtype in array_specs:
      setattr(self, name, _allocate((n_inputs, initial_n_segments, width), dtype))
    self._logger.debug(f"Resulting array allocated: {self.memory_usage_mb:.2f} MB")

  @property
  def memory_usage_mb(self) -> float:
    return sum(array.nbytes for array in self.arrays()) / 1024**2

  @property
  def current_n_segments(self) -> int:
    return int(self.arrays()[0].shape[1])

  def arrays(self) -> tuple[np.ndarray, ...]:
    """Return the arrays themselves (not copies) in ``_ARRAYS`` order."""
    return tuple(getattr(self, name) for name, _ in self._ARRAYS)

  def _ensure_capacity(self, needed_n_segments: int) -> None:
    if needed_n_segments <= self.current_n_segments:
      return

    old_n_segments = self.current_n_segments

    for name, _ in self._ARRAYS:
      old = getattr(self, name)
      new = _allocate((old.shape[0], needed_n_segments, old.shape[2]), old.dtype)
      new[:, :old_n_segments, :] = old
      setattr(self, name, new)

    self._logger.debug(
      f"[resized] from {old_n_segments} to {needed_n_segments} segments. "
      f"Resulting array allocated: {self.memory_usage_mb:.2f} MB"
    )

  def _prepare_block(
    self, input_indices: np.ndarray, segment_indices: np.ndarray
  ) -> None:
    """Check the index dtypes and grow the arrays to hold ``segment_indices``."""
    assert input_indices.dtype == self._input_indices_dtype
    assert segment_indices.dtype == self._segment_indices_dtype
    block_max_segment_idx = segment_indices.max()
    max_segment_size = max(block_max_segment_idx, self._max_segment_index.value) + 1
    self._ensure_capacity(max_segment_size)

  def set_unprocessable_inputs(self, unprocessable_inputs: set[int]) -> None:
    super().set_unprocessable_inputs(unprocessable_inputs)
    for name, value in self._ARRAYS:
      getattr(self, name)[self._unprocessable_inputs, :, :] = value

  def copy_file_slice(self, file_idx: int, n_segments: int) -> tuple[np.ndarray, ...]:
    assert 0 <= n_segments <= self.current_n_segments
    return tuple(
      array[file_idx, :n_segments].copy()[np.newaxis] for array in self.arrays()
    )
