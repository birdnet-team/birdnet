from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import os
import shutil
import tempfile
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager, suppress
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
from birdnet.utils.download_progress import (
  DownloadReporter,
  get_download_progress_callback,
)


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


# Written into a downloaded SavedModel directory to record which release it came
# from. The directory name is generic, so unlike the size-checked single-file
# models a directory cached from an older release is otherwise indistinguishable
# from a current one and would never be re-downloaded on upgrade.
SOURCE_MARKER_NAME = ".birdnet_source"


def write_source_marker(model_dir: Path, dl_url: str) -> None:
  (model_dir / SOURCE_MARKER_NAME).write_text(dl_url, encoding="utf-8")


def check_source_marker(model_dir: Path, dl_url: str) -> bool:
  marker = model_dir / SOURCE_MARKER_NAME
  if not marker.is_file():
    return False
  return marker.read_text(encoding="utf-8").strip() == dl_url


@dataclass()
class ModelInfo:
  dl_url: str
  dl_size: int
  file_size: int
  dl_file_name: str
  # Unset only for the zip-packaged v2.4 downloads, which are still judged by
  # extracted byte size alone.
  sha256: str | None = None

  @property
  def content_tag(self) -> str:
    """Checksum prefix embedded in the cached file's name.

    The name then identifies the release content itself: a release that expects
    different bytes looks for a different file, so it can neither serve another
    release's model nor overwrite it in a shared app data directory."""
    assert self.sha256 is not None
    return self.sha256[:12]


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


def write_text_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
  """Write via a temporary file in the same directory, then rename over.

  A reader either sees the previous content or the new one, never a partial
  file, and an interrupted write leaves nothing behind but a `.tmp` scratch file.

  Newlines are written through unchanged, so the same content yields the same
  bytes on every platform. Without that, files generated on Windows and on Linux
  differ, and anything that records a checksum of them records a different one
  per operating system.
  """
  fd, temp_name = tempfile.mkstemp(
    dir=path.parent,
    prefix=f"{path.name}.",
    suffix=".tmp",
  )
  os.close(fd)
  temp_path = Path(temp_name)

  try:
    temp_path.write_text(content, encoding=encoding, newline="\n")
    temp_path.replace(path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise


_LOCK_OWNER_NAME = "owner"

# A lock whose owner never wrote its pid is only reclaimed once it is old enough
# that the owner cannot still be between mkdir and the write.
_LOCK_ADOPTION_GRACE_S = 30.0


def _reclaim_if_abandoned(lock_dir: Path) -> bool:
  """Drop a lock whose owner is gone. Returns whether anything was reclaimed.

  Without this a process killed mid-setup - force-quit, OOM, power loss - leaves
  the directory behind and every later call waits out the timeout and fails, for
  good, with nothing to do but delete a hidden directory by hand.
  """
  import psutil

  owner = lock_dir / _LOCK_OWNER_NAME
  try:
    pid = int(owner.read_text(encoding="utf-8").strip())
  except (OSError, ValueError):
    try:
      age = time.time() - lock_dir.stat().st_mtime
    except OSError:
      return False
    if age < _LOCK_ADOPTION_GRACE_S:
      return False
  else:
    if psutil.pid_exists(pid):
      return False

  shutil.rmtree(lock_dir, ignore_errors=True)
  return not lock_dir.exists()


@contextmanager
def directory_lock(
  lock_dir: Path, description: str, timeout_s: float = 300.0
) -> Generator[None, None, None]:
  """Serialize one-time setup across processes by creating a directory.

  `mkdir` is atomic on every platform this runs on, which a lock file is not.
  The holder records its pid inside, so a lock left behind by a process that no
  longer exists is reclaimed rather than waited out.
  """
  deadline = time.monotonic() + timeout_s
  while True:
    try:
      lock_dir.mkdir(parents=True, exist_ok=False)
      break
    except FileExistsError as err:
      if _reclaim_if_abandoned(lock_dir):
        continue
      if time.monotonic() >= deadline:
        raise TimeoutError(
          f"Timed out while waiting for {description}. Another process is "
          f"holding {lock_dir}; if none is running, remove that directory."
        ) from err
      time.sleep(0.1)

  with suppress(OSError):
    (lock_dir / _LOCK_OWNER_NAME).write_text(str(os.getpid()), encoding="utf-8")
  try:
    yield
  finally:
    shutil.rmtree(lock_dir, ignore_errors=True)


def get_supported_audio_files_recursive(folder: Path) -> Generator[Path, None, None]:
  assert folder.is_dir()
  yield from (
    p.absolute()
    for p in folder.rglob("*")
    if p.is_file() and is_supported_audio_file(p)
  )


def assert_queue_is_empty(queue: Queue) -> None:
  # qsize() doesn't work on macOS
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


def flat_softmax_fast(x: npt.NDArray) -> npt.NDArray:
  x_max = np.max(x, axis=1, keepdims=True)
  shifted = x - x_max
  exp_shifted = np.exp(shifted, dtype=x.dtype)
  return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


# Transient network faults (connection resets, read timeouts, truncated
# streams, 5xx) must not fail a model/label download outright: every official
# model goes through this helper, so a single fault would otherwise surface as
# a failed `load()`. Client errors (4xx) are permanent and are raised at once.
#
# The back-off spans ~110 s in total because the observed failure mode is not a
# single dropped packet: GitHub's release-download endpoint refuses connections
# ("Remote end closed connection without response") for tens of seconds at a
# time. A 20 s window was measured in CI to be too short -- a sibling step
# retrying the same host after 30 s succeeded on its second try while this
# helper exhausted three attempts. The wait is only ever paid on a download
# that is already failing.
_DOWNLOAD_ATTEMPTS = 5
_DOWNLOAD_RETRY_WAITS_S = (5.0, 15.0, 30.0, 60.0)


class DownloadError(ValueError):
  """A download did not complete successfully.

  Subclasses ``ValueError`` because that is what this helper has always raised
  for a failed download; ``status_code`` is exposed so callers (and the retry
  loop below) can tell a permanent client error from a retriable one.
  """

  def __init__(self, message: str, *, status_code: int | None = None) -> None:
    super().__init__(message)
    self.status_code = status_code


def _is_retriable_download_error(error: Exception) -> bool:
  import requests

  status: int | None = None
  if isinstance(error, DownloadError):
    status = error.status_code
  elif isinstance(error, requests.HTTPError) and error.response is not None:
    status = error.response.status_code

  # 4xx means we asked for something that does not exist or may not be read;
  # repeating the request cannot change that. Everything else -- connection
  # resets, timeouts, truncated streams, 5xx -- is worth another attempt.
  is_client_error = status is not None and 400 <= status < 500
  return not is_client_error


def download_file_tqdm(
  url: str,
  file_path: Path,
  *,
  download_size: int | None = None,
  description: str | None = None,
) -> int:
  import requests

  # The callback is captured once per download, so a registration that changes
  # mid-way takes effect from the next download on (and the tqdm bar's on/off
  # state stays consistent with the events for this one).
  reporter = DownloadReporter(
    url, description, _DOWNLOAD_ATTEMPTS, get_download_progress_callback()
  )
  attempt = 0
  try:
    while True:
      attempt += 1
      try:
        return _download_file_once(
          url,
          file_path,
          download_size=download_size,
          description=description,
          attempt=attempt,
          reporter=reporter,
        )
      except (requests.RequestException, ValueError) as error:
        # Re-raise from inside the handler so the original traceback survives.
        if reporter.callback_failed:
          # The callback raised (a ValueError lands here too): abort, no retry.
          raise
        if attempt >= _DOWNLOAD_ATTEMPTS or not _is_retriable_download_error(error):
          raise
        wait_s = _DOWNLOAD_RETRY_WAITS_S[
          min(attempt - 1, len(_DOWNLOAD_RETRY_WAITS_S) - 1)
        ]
        # Before the log line: a callback that raises here cancels the download,
        # and then nothing is retried.
        reporter.retrying(error, wait_s)
        logging.getLogger(__name__).warning(
          f"Download of {url} failed (attempt {attempt}/{_DOWNLOAD_ATTEMPTS}): "
          f"{error}. Retrying in {wait_s:.0f} s..."
        )
        time.sleep(wait_s)
  except BaseException as error:
    # Terminal event for errors the retry handler does not deal in (OSError,
    # KeyboardInterrupt). A raising callback gets no further events; suppress
    # keeps one that raises again from masking the original error.
    if not reporter.callback_failed:
      with suppress(BaseException):
        reporter.failed(error)
    raise


def _download_file_once(
  url: str,
  file_path: Path,
  *,
  download_size: int | None = None,
  description: str | None = None,
  attempt: int = 1,
  reporter: DownloadReporter | None = None,
) -> int:
  assert file_path.parent.is_dir()
  import requests

  if reporter is None:
    reporter = DownloadReporter(url, description, _DOWNLOAD_ATTEMPTS, callback=None)

  reporter.started(attempt, download_size)

  response = requests.get(url, stream=True, timeout=120)
  try:
    content_length = response.headers.get("content-length")
    if download_size is not None:
      total_size = download_size
    elif content_length is not None:
      total_size = int(content_length)
    else:
      total_size = 0
    # 0 is the internal "unknown" sentinel (it also disables the size check).
    bytes_total: int | None = total_size
    if download_size is None and content_length is None:
      bytes_total = None
    reporter.total_known(bytes_total)

    block_size = 1024
    fd, temp_name = tempfile.mkstemp(
      dir=file_path.parent,
      prefix=f"{file_path.name}.",
      suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)

    downloaded_size = 0
    try:
      with (
        tqdm(
          total=total_size,
          unit="iB",
          unit_scale=True,
          desc=description,
          disable=reporter.enabled,
        ) as tqdm_bar,
        open(temp_path, "wb") as file,
      ):
        for data in response.iter_content(block_size):
          # Bytes are counted here rather than read from `tqdm_bar.n`, which
          # does not advance while the bar is disabled.
          tqdm_bar.update(len(data))
          file.write(data)
          downloaded_size += len(data)
          reporter.progress(downloaded_size)

      if response.status_code != 200 or (total_size not in (0, downloaded_size)):
        raise DownloadError(
          f"Failed to download the file. Status code: {response.status_code}\n"
          f"Expected size: {total_size} bytes, "
          f"downloaded size: {downloaded_size} bytes.",
          status_code=response.status_code,
        )

      temp_path.replace(file_path)
    except BaseException:
      # BaseException: a KeyboardInterrupt, or a callback raising to cancel,
      # must not leave the partial file behind either.
      temp_path.unlink(missing_ok=True)
      raise
  finally:
    response.close()

  reporter.finished()
  return total_size


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def ensure_single_file_model(
  info: ModelInfo, model_path: Path, legacy_path: Path, description: str
) -> None:
  """Put the model file `info` describes at `model_path`, downloading if needed.

  `model_path` carries `info.content_tag` and only verified bytes are ever
  renamed onto it, so its presence at the declared byte size means the file
  this release published; nothing is hashed again on later loads, which would
  cost seconds for the ~540 MB acoustic models. Byte size alone could not give
  that guarantee: a retrain that keeps the architecture keeps the tflite/onnx
  size too — the size check told the geo v3.0.3 and v3.0.4 caches apart only
  because their class counts differ.

  A matching file left at `legacy_path` by a release before names carried the
  checksum is adopted rather than fetched again, so upgrading stays offline for
  anyone already holding it. One that hashes differently is left alone: it
  belongs to another installed version still reading it from there.
  """
  assert info.sha256 is not None
  if model_path.is_file() and model_path.stat().st_size == info.file_size:
    return

  model_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    if (
      legacy_path.is_file()
      and legacy_path.stat().st_size == info.file_size
      and sha256_file(legacy_path) == info.sha256
    ):
      # Windows refuses this while another process has the file open; the
      # re-check below then sends this process to the download instead.
      os.replace(legacy_path, model_path)
      return
  except OSError:
    # The legacy file vanished mid-probe: a concurrent process adopted it.
    pass
  # Downloads are not serialized, so a concurrent process may have adopted or
  # downloaded the file while this one was probing; its bytes are the wanted
  # ones (only verified content ever lands at this path).
  if model_path.is_file() and model_path.stat().st_size == info.file_size:
    return

  # Downloaded next to the target and renamed only after verification, so a
  # process killed during the hash cannot leave bytes nobody checked at a name
  # every later load trusts. The scratch name is per-process; a shared one
  # would let concurrent downloads fail each other's rename on Windows.
  fd, temp_name = tempfile.mkstemp(
    dir=model_path.parent,
    prefix=f"{model_path.name}.",
    suffix=".unverified",
  )
  os.close(fd)
  temp_path = Path(temp_name)
  try:
    download_file_tqdm(
      info.dl_url, temp_path, download_size=info.dl_size, description=description
    )
    actual = sha256_file(temp_path)
    if actual != info.sha256:
      raise RuntimeError(
        f"The file downloaded from {info.dl_url} does not match its expected "
        f"checksum ({actual} instead of {info.sha256}). It was discarded; "
        "retry, and if this persists the published file has changed."
      )
    try:
      temp_path.replace(model_path)
    except OSError:
      # Windows refuses this while a concurrent process still holds the file it
      # just published open. Anything already at this path carries the same
      # content tag, so those are the wanted bytes and this one's are redundant.
      if not (model_path.is_file() and model_path.stat().st_size == info.file_size):
        raise
  finally:
    temp_path.unlink(missing_ok=True)


def itertools_batched(iterable: Iterable, n: int) -> Generator[Any, None, None]:
  # https://docs.python.org/3.12/library/itertools.html#itertools.batched
  # batched('ABCDEFG', 3) → ABC DEF G
  if n < 1:
    raise ValueError("n must be at least one")
  iterator = iter(iterable)
  while batch := tuple(islice(iterator, n)):
    yield batch
