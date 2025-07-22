from __future__ import annotations

import ctypes
import logging
import math
import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

from birdnet.logging_utils import get_logger

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter


def check_protobuf_model_files_exist(folder: Path) -> bool:
  exists = True
  exists &= (folder / "saved_model.pb").is_file()
  exists &= (folder / "variables").is_dir()
  exists &= (folder / "variables" / "variables.data-00000-of-00001").is_file()
  exists &= (folder / "variables" / "variables.index").is_file()
  return exists


def load_pb_model(model_path: Path):
  import absl.logging

  absl_verbosity_before = absl.logging.get_verbosity()
  absl.logging.set_verbosity(absl.logging.ERROR)
  tf_verbosity_before = logging.getLogger("tensorflow").level
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  import tensorflow as tf

  # Note: memory growth needs to be set before loading the model and maybe only once in the main process
  # physical_gpu_device = gpus_with_name[0]
  # if tf.config.experimental.get_memory_growth(physical_gpu_device) is False:
  #   tf.config.experimental.set_memory_growth(physical_gpu_device, True)

  start = time.perf_counter()
  model = tf.saved_model.load(str(model_path.absolute()))
  end = time.perf_counter()
  logger = get_logger(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} in {end - start:.2f} seconds."
  )

  absl.logging.set_verbosity(absl_verbosity_before)
  logging.getLogger("tensorflow").setLevel(tf_verbosity_before)
  return model


def load_tf_model(
  model_path: Path,
  allocate_tensors: bool = False,
) -> TFInterpreter:
  assert model_path.is_file()
  assert tf_installed()

  absl_verbosity_before: int | None = None
  tf_verbosity_before: int | None = None

  import absl.logging as absl_logging

  absl_verbosity_before = absl_logging.get_verbosity()
  absl_logging.set_verbosity(absl_logging.ERROR)
  absl_logging.set_stderrthreshold("error")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  tf_verbosity_before: int | None = None
  tf_verbosity_before = logging.getLogger("tensorflow").level
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  # NOTE: import in this way is not possible:
  # `import tensorflow.lite.python.interpreter as tflite`
  from tensorflow.lite.python import interpreter as tflite

  # memory_map not working for TF 2.15.1:
  # f = open(self._model_path, "rb")
  # self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  start = time.perf_counter()
  try:
    interp = tflite.Interpreter(
      str(model_path.absolute()),
      num_threads=1,
      experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,  # tensor#187 is a dynamic-sized tensor # type: ignore
    )
  except ValueError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'tensorflow'. Ensure it is a valid TFLite model."
    ) from e

  end = time.perf_counter()
  logger = get_logger(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'tensorflow' in {end - start:.2f} seconds."
  )

  if allocate_tensors:
    interp.allocate_tensors()

  import absl.logging as absl_logging

  assert absl_verbosity_before is not None
  assert tf_verbosity_before is not None
  absl_logging.set_verbosity(absl_verbosity_before)
  logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

  return interp


def load_litert_model(
  model_path: Path,
  allocate_tensors: bool = False,
) -> LiteRTInterpreter:
  assert model_path.is_file()
  assert litert_installed()

  from ai_edge_litert import interpreter as tflite

  start = time.perf_counter()
  try:
    interp = tflite.Interpreter(
      str(model_path.absolute()),
      num_threads=1,
      experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,  # tensor#187 is a dynamic-sized tensor # type: ignore
    )
  except ValueError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'ai_edge_litert'. Ensure it is a valid TFLite model."
    ) from e

  end = time.perf_counter()
  logger = get_logger(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'ai_edge_litert' in {end - start:.2f} seconds."
  )

  if allocate_tensors:
    interp.allocate_tensors()

  return interp


def tf_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("tensorflow") is not None


def litert_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("ai_edge_litert") is not None


@dataclass()
class ModelInfo:
  dl_url: str
  dl_size: int
  file_size: int
  dl_file_name: str


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
    with suppress(FileNotFoundError):
      shm.unlink()
    logger = logging.getLogger(__name__)
    logger.debug(f"Shared memory {ring.name} cleaned up.")


def get_max_n_segments(
  max_duration_s: float, segment_size_s: float, overlap_duration_s: float
) -> int:
  effective_segment_duration_s = segment_size_s - overlap_duration_s
  assert effective_segment_duration_s > 0
  n_segments = math.ceil(max_duration_s / effective_segment_duration_s)
  return n_segments


def get_max_n_segments_array(
  max_duration_s: np.ndarray, segment_size_s: float, overlap_duration_s: float
) -> np.ndarray:
  max_val = get_max_n_segments(
    np.max(max_duration_s), segment_size_s, overlap_duration_s
  )
  dtype = uint_dtype_for(max_val)

  effective_segment_duration_s = segment_size_s - overlap_duration_s
  assert effective_segment_duration_s > 0
  n_segments = np.ceil(max_duration_s / effective_segment_duration_s).astype(dtype)
  return n_segments


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


def uint_ctype_from_dtype(
  dtype: DTypeLike,
) -> ctypes.c_uint8 | ctypes.c_uint16 | ctypes.c_uint32 | ctypes.c_uint64:
  dtype = np.dtype(dtype).type  # z. B. <class 'numpy.uint16'>
  code = _UINT_DTYPE_TO_CTYPE[dtype]
  return code
