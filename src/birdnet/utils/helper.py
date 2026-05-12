from __future__ import annotations

import ctypes
import hashlib
import math
import os
import tempfile
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import DTypeLike
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.globals import Float32Array


def format_input_for_csv(input_value: Any) -> str:  # noqa: ANN401
  return f'"{input_value}"'


def hms_centis_fast(v: float) -> str:
  h, rem = divmod(v, 3600)
  m, s = divmod(rem, 60)
  result = f"{int(h):02}:{int(m):02}:{s:05.2f}"
  return result


def check_is_intel_macos() -> bool:
  import platform

  if platform.system() == "Darwin":
    is_intel = platform.machine() == "x86_64"
    return is_intel
  return False


def check_is_python_312() -> bool:
  import sys

  return sys.version_info.major == 3 and sys.version_info.minor == 12


def get_hash(session_id: str) -> str:
  hash_digest = hashlib.sha256(session_id.encode()).hexdigest()
  return hash_digest


def check_protobuf_model_files_exist(folder: Path) -> bool:
  exists = True
  exists &= (folder / "saved_model.pb").is_file()
  exists &= (folder / "variables").is_dir()
  exists &= (folder / "variables" / "variables.data-00000-of-00001").is_file()
  exists &= (folder / "variables" / "variables.index").is_file()
  return exists


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
# Not supported: {".AAC", ".WMA", ".M4A"}


# ---------------- Mapping -----------------
_UINT_DTYPE_TO_CTYPE = {
  np.uint8: ctypes.c_uint8,
  np.uint16: ctypes.c_uint16,
  np.uint32: ctypes.c_uint32,
  np.uint64: ctypes.c_uint64,
}


def get_supported_audio_files_recursive(folder: Path) -> Generator[Path, None, None]:
  assert folder.is_dir()
  yield from (
    p.absolute()
    for p in folder.rglob("*")
    if p.is_file() and is_supported_audio_file(p)
  )


def assert_queue_is_empty(queue: Queue) -> None:
  # qsize() doesn't work on macOS:
  # assert self._files_queue.qsize() == 0
  try:
    queue.get_nowait()
    raise AssertionError("Queue is not empty!")
  except Empty:
    pass


def is_supported_audio_file(file_path: Path) -> bool:
  assert file_path.is_file()
  return file_path.suffix.upper() in SF_FORMATS


def validate_species_list(species_list: Path) -> OrderedSet[str]:
  loaded_species_list: OrderedSet[str]
  try:
    loaded_species_list = get_species_from_file(species_list, encoding="utf8")
  except Exception as e:
    raise ValueError(
      f"Failed to read species list from '{species_list.absolute()}'. "
      f"Ensure it is a valid text file."
    ) from e

  if len(loaded_species_list) == 0:
    raise ValueError(f"Species list '{species_list.absolute()}' is empty!")

  return loaded_species_list


def max_value_for_uint_dtype(dtype: np.dtype) -> int:
  """
  Returns the maximum value that can be represented by the given NumPy dtype.
  """
  assert np.issubdtype(dtype, np.integer)
  return np.iinfo(dtype).max


def xget_max_n_segments(
  max_duration_s: float, segment_size_s: float, overlap_duration_s: float
) -> int:
  effective_segment_duration_s = segment_size_s - overlap_duration_s
  assert effective_segment_duration_s > 0
  n_segments = math.ceil(max_duration_s / effective_segment_duration_s)
  return n_segments


def apply_speed_to_duration(duration_s: float, speed: float) -> float:
  assert speed > 0
  scaled_duration = duration_s * speed
  return scaled_duration


def apply_speed_to_samples(samples: int, speed: float) -> int:
  assert speed > 0
  scaled_samples = round(samples * speed)
  return scaled_samples


def get_hop_duration_s(
  segment_size_s: float, overlap_duration_s: float, speed: float
) -> float:
  assert speed > 0
  assert segment_size_s > overlap_duration_s
  hop_duration_s = apply_speed_to_duration(segment_size_s - overlap_duration_s, speed)
  assert hop_duration_s > 0
  return hop_duration_s


def get_n_segments_speed(
  duration_s: float, segment_size_s: float, overlap_duration_s: float, speed: float
) -> int:
  hop_duration_s = get_hop_duration_s(segment_size_s, overlap_duration_s, speed)
  n_segments = math.ceil(duration_s / hop_duration_s)
  return n_segments


def duration_as_samples(duration_s: float, sample_rate: int) -> int:
  return round(duration_s * sample_rate)


def uint_ctype_from_dtype(
  dtype: DTypeLike,
) -> ctypes.c_uint8 | ctypes.c_uint16 | ctypes.c_uint32 | ctypes.c_uint64:
  dtype = np.dtype(dtype).type  # z. B. <class 'numpy.uint16'>
  code = _UINT_DTYPE_TO_CTYPE[dtype]
  return code


def uint_dtype_for_files(n_files: int) -> np.dtype:
  return get_uint_dtype(n_files - 1)


def get_uint_dtype(max_value: int) -> np.dtype:
  """
  Return the narrowest unsigned-integer NumPy dtype that can represent
  *max_value* (inclusive).

  Examples
  --------
  >>> get_uint_dtype(100)
  dtype('uint8')
  >>> get_uint_dtype(42_000)
  dtype('uint16')
  >>> get_uint_dtype(3_000_000_000)
  dtype('uint64')

  Notes
  -----
  2**8 = 256
  2**16 = 65,536
  2**32 = 4,294,967,296
  2**64 = 18,446,744,073,709,551,616
  """
  assert max_value >= 0, "max_value must be non-negative."

  for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
    if max_value <= np.iinfo(dt).max:
      return np.dtype(dt)

  raise AssertionError("Value exceeds uint64 range.")


def get_float_dtype(max_value: float) -> DTypeLike:
  """
  Magnitude-based: returns the smallest float dtype whose range covers max_value.
  Use for bulk arrays where memory matters and per-element
  rounding is acceptable (e.g. lists of file durations).
  """
  if max_value <= 2**11:
    return np.float16
  elif max_value <= 2**24:
    return np.float32
  else:
    return np.float64


def upgrade_float_dtype_for_value(dtype: np.dtype, value: float) -> np.dtype:
  if dtype == np.float16 and float(np.float16(value)) != float(value):
    dtype = np.dtype(np.float32)
  if dtype == np.float32 and float(np.float32(value)) != float(value):
    dtype = np.dtype(np.float64)
  return dtype


# Lossless: smallest float dtype that represents value exactly. Use for scalar
# configuration parameters (speed, segment/overlap duration) where the value
# feeds into derived computations and rounding accumulates over many segments.
def get_lossless_float_dtype(value: float) -> np.dtype:
  return upgrade_float_dtype_for_value(np.dtype(get_float_dtype(value)), value)


def get_file_formats(file_paths: set[Path]) -> str:
  return ", ".join(sorted({x.suffix[1:].upper() for x in file_paths}))


def get_species_from_file(
  species_file: Path, /, *, encoding: str = "utf8"
) -> OrderedSet[str]:
  species = OrderedSet(species_file.read_text(encoding).strip().splitlines())
  return species


def bandpass_signal(
  audio_signal: Float32Array,
  rate: int,
  fmin: int,
  fmax: int,
  new_fmin: int,
  new_fmax: int,
) -> Float32Array:
  assert rate > 0
  assert fmin >= 0
  assert fmin < fmax
  assert new_fmin >= 0
  assert new_fmin < new_fmax

  from scipy.signal import butter, lfilter

  nth_order = 5
  nyquist = rate // 2

  # Highpass
  if fmin > new_fmin and fmax == new_fmax:
    low = fmin / nyquist
    b, a = butter(nth_order, low, btype="high")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  # Lowpass
  elif fmin == new_fmin and fmax < new_fmax:
    high = fmax / nyquist
    b, a = butter(nth_order, high, btype="low")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  # Bandpass
  elif fmin > new_fmin and fmax < new_fmax:
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(nth_order, [low, high], btype="band")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  sig_f32 = audio_signal.astype(np.float32)
  return sig_f32


def fillup_with_silence(
  audio_segment: Float32Array, target_length: int
) -> Float32Array:
  current_length = len(audio_segment)
  assert current_length <= target_length

  if current_length == target_length:
    return audio_segment

  silence_length = target_length - current_length
  silence = np.zeros(silence_length, dtype=audio_segment.dtype)
  filled_segment = np.concatenate((audio_segment, silence))

  return filled_segment


def flat_sigmoid_logaddexp_fast(
  x: npt.NDArray, sensitivity: float, clip_val: float = 15.0, bias: float = 1.0
) -> npt.NDArray:
  transformed_bias = (bias - 1.0) * 10.0
  y = sensitivity * np.clip(x + transformed_bias, -clip_val, clip_val)

  positive_mask = y >= 0
  abs_y = np.abs(y)
  exp_neg_abs = np.exp(-abs_y, dtype=x.dtype)

  one_plus_exp = 1.0 + exp_neg_abs

  return np.where(positive_mask, exp_neg_abs / one_plus_exp, 1.0 / one_plus_exp)


def download_file_tqdm(
  url: str,
  file_path: Path,
  *,
  download_size: int | None = None,
  description: str | None = None,
) -> int:
  assert file_path.parent.is_dir()
  import requests

  response = requests.get(url, stream=True, timeout=120)
  total_size = int(response.headers.get("content-length", 0))
  if download_size is not None:
    total_size = download_size

  block_size = 1024
  fd, temp_name = tempfile.mkstemp(
    dir=file_path.parent,
    prefix=f"{file_path.name}.",
    suffix=".tmp",
  )
  os.close(fd)
  temp_path = Path(temp_name)

  try:
    with (
      tqdm(total=total_size, unit="iB", unit_scale=True, desc=description) as tqdm_bar,
      open(temp_path, "wb") as file,
    ):
      for data in response.iter_content(block_size):
        tqdm_bar.update(len(data))
        file.write(data)

    if response.status_code != 200 or (total_size not in (0, tqdm_bar.n)):
      raise ValueError(
        f"Failed to download the file. Status code: {response.status_code}"
      )

    temp_path.replace(file_path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise
  finally:
    response.close()

  return total_size


def itertools_batched(iterable: Iterable, n: int) -> Generator[Any, None, None]:
  # https://docs.python.org/3.12/library/itertools.html#itertools.batched
  # batched('ABCDEFG', 3) → ABC DEF G
  if n < 1:
    raise ValueError("n must be at least one")
  iterator = iter(iterable)
  while batch := tuple(islice(iterator, n)):
    yield batch
