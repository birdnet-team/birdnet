import ctypes
import logging
import math
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Generator

import numpy as np
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

from birdnet.logging_utils import get_logger

SF_FORMATS = {
  ".AIFC",
  ".AIFF",
  ".AU",
  ".AVR",
  ".CAF",
  ".FLAC",
  ".HTK",
  ".IRCAM",
  ".MAT4",
  ".MAT5",
  ".MP3",
  ".MPC2K",
  ".NIST",
  ".OGG",
  ".OPUS",
  ".PAF",
  ".PVF",
  ".RAW",
  ".RF64",
  ".SD2",
  ".SDS",
  ".SVX",
  ".VOC",
  ".W64",
  ".WAV",
  ".WAVEX",
  ".WVE",
  ".XI",
}


def get_supported_audio_files(folder: Path) -> Generator[Path, None, None]:
  assert folder.is_dir()
  result = (
    p.absolute() for p in folder.rglob("**/*") if p.suffix.upper() in SF_FORMATS
  )
  yield from result


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

  def cleanup(self) -> None:
    try:
      shm = self.attach_shared_memory()
    except FileNotFoundError:
      return
    else:
      logger = get_logger(__name__)
      logger.debug(f"Cleaning up shared memory {self.name}.")
      shm.close()
      with suppress(FileNotFoundError):
        shm.unlink()
      logger.debug(f"Shared memory {self.name} cleaned up.")

  def get_array(self, shm: shared_memory.SharedMemory) -> np.ndarray:
    view = np.ndarray(self.shape, self.dtype, buffer=shm.buf)
    return view

  def attach_and_get_array(self) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = self.attach_shared_memory()
    view = self.get_array(shm)
    return shm, view


@contextmanager  # type: ignore
def create_shm_ring(ring: RingField) -> shared_memory.SharedMemory:  # type: ignore
  shm = shared_memory.SharedMemory(name=ring.name, create=True, size=ring.nbytes)
  try:
    yield shm  # type: ignore
  finally:
    shm.close()
    shm.unlink()
    logger = logging.getLogger(__name__)
    logger.debug(f"Shared memory {ring.name} cleaned up.")


def get_max_n_chunks(
  max_duration_s: float, chunk_size_s: float, overlap_duration_s: float
) -> int:
  effective_chunk_duration_s = chunk_size_s - overlap_duration_s
  assert effective_chunk_duration_s > 0
  n_chunks = math.ceil(max_duration_s / effective_chunk_duration_s)
  return n_chunks


# ---------------- Mapping -----------------
_DTYPE_TO_CODE = {
  np.uint8: "B",  # unsigned char
  np.int8: "b",
  np.uint16: "H",  # unsigned short
  np.int16: "h",
  np.uint32: "I",  # unsigned int
  np.int32: "i",
  np.uint64: "Q",  # unsigned long long
  np.int64: "q",
  np.float32: "f",
  np.float64: "d",
}

# ---------------- Mapping -----------------
_UINT_DTYPE_TO_CTYPE = {
  np.uint8: ctypes.c_uint8,
  np.uint16: ctypes.c_uint16,
  np.uint32: ctypes.c_uint32,
  np.uint64: ctypes.c_uint64,
}


def code_from_dtype(dtype: DTypeLike) -> str:
  dtype = np.dtype(dtype).type  # z. B. <class 'numpy.uint16'>
  code = _DTYPE_TO_CODE[dtype]
  return code


def uint_ctype_from_dtype(dtype: DTypeLike) -> ctypes._SimpleCData:
  dtype = np.dtype(dtype).type  # z. B. <class 'numpy.uint16'>
  code = _UINT_DTYPE_TO_CTYPE[dtype]
  return code
