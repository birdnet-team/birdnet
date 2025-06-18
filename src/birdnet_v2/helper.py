import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


def uint_dtype_for_files(n_files: int) -> np.dtype:
  return uint_dtype_for(n_files - 1)


def uint_dtype_for(max_value: int) -> np.dtype:
  """
  Return the narrowest unsigned-integer NumPy dtype that can represent
  *max_value* (inclusive).

  Examples
  --------
  >>> uint_dtype_for(100)
  dtype('uint8')
  >>> uint_dtype_for(42_000)
  dtype('uint16')
  >>> uint_dtype_for(3_000_000_000)
  dtype('uint64')
  """
  assert max_value >= 0, "max_value must be non-negative."

  for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
    if max_value <= np.iinfo(dt).max:
      return np.dtype(dt)

  raise AssertionError("Value exceeds uint64 range.")


def max_value_for_uint_dtype(dtype: np.dtype) -> int:
  """
  Returns the maximum value that can be represented by the given NumPy dtype.
  """
  assert np.issubdtype(dtype, np.integer)
  return np.iinfo(dtype).max


@dataclass(slots=True, frozen=True)
class RingField:
  name: str
  dtype: np.dtype
  shape: tuple[int, ...]

  # ----------------------------------------
  @property
  def nbytes(self) -> int:
    return int(np.prod(self.shape)) * self.dtype.itemsize

  def attach_shared_memory(self) -> shared_memory.SharedMemory:
    """
    Attaches to an existing shared memory segment with the specified name.
    """
    return shared_memory.SharedMemory(name=self.name, create=False)

  def get_array(self, shm: shared_memory.SharedMemory) -> np.ndarray:
    view = np.ndarray(self.shape, self.dtype, buffer=shm.buf)
    return view

  def attach_and_get_array(self) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = self.attach_shared_memory()
    view = self.get_array(shm)
    return shm, view


@contextmanager  # type: ignore
def create_shm_ring(ring: RingField) -> shared_memory.SharedMemory:  # type: ignore
  shm = shared_memory.SharedMemory(create=True, name=ring.name, size=ring.nbytes)
  try:
    yield shm  # type: ignore
  finally:
    shm.close()
    shm.unlink()  # wird sogar bei CTRL-C im finally ausgeführt
    logger = logging.getLogger(__name__)
    logger.debug(f"Shared memory {ring.name} cleaned up.")


def get_max_n_chunks(
  max_duration_min: float, chunk_size_s: float, overlap_duration_s: float
) -> int:
  total_duration_s = max_duration_min * 60
  effective_chunk_duration_s = chunk_size_s - overlap_duration_s
  assert effective_chunk_duration_s > 0
  n_chunks = math.ceil(total_duration_s / effective_chunk_duration_s)
  return n_chunks
