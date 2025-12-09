import contextlib
import ctypes
import importlib.util
import multiprocessing as mp
import os
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from multiprocessing import get_all_start_methods, set_start_method
from typing import IO

import numpy as np
import psutil
import pytest

from birdnet.acoustic_models.inference.encoding.result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic_models.inference.prediction.result import (
  AcousticPredictionResultBase,
)
from birdnet.backends import litert_installed


def create_zero_len_wav(f: IO[bytes]) -> None:
  # 44-Byte WAV Header
  f.write(
    b"RIFF"
    + (36).to_bytes(4, "little")
    + b"WAVEfmt "
    + (16).to_bytes(4, "little")  # fmt chunk length
    + (1).to_bytes(2, "little")  # PCM
    + (1).to_bytes(2, "little")  # channels
    + (44100).to_bytes(4, "little")  # sample rate
    + (88200).to_bytes(4, "little")  # byte rate
    + (2).to_bytes(2, "little")  # block align
    + (16).to_bytes(2, "little")  # bits per sample
    + b"data"
    + (0).to_bytes(4, "little")  # data size = 0
  )


def assert_encoding_result_is_equal(
  a: AcousticEncodingResultBase, b: AcousticEncodingResultBase
) -> None:
  assert isinstance(a, AcousticEncodingResultBase)
  assert isinstance(b, AcousticEncodingResultBase)

  np.testing.assert_array_equal(a.segment_duration_s, b.segment_duration_s)
  np.testing.assert_array_equal(a.overlap_duration_s, b.overlap_duration_s)
  np.testing.assert_array_equal(a.inputs, b.inputs)
  np.testing.assert_array_equal(a.input_durations, b.input_durations)
  np.testing.assert_array_equal(a.embeddings, b.embeddings)
  np.testing.assert_array_equal(a.embeddings_masked, b.embeddings_masked)


def assert_prediction_result_is_equal(
  a: AcousticPredictionResultBase, b: AcousticPredictionResultBase
) -> None:
  assert isinstance(a, AcousticPredictionResultBase)
  assert isinstance(b, AcousticPredictionResultBase)

  np.testing.assert_array_equal(a.segment_duration_s, b.segment_duration_s)
  np.testing.assert_array_equal(a.overlap_duration_s, b.overlap_duration_s)
  np.testing.assert_array_equal(a.inputs, b.inputs)
  np.testing.assert_array_equal(a.species_list, b.species_list)
  np.testing.assert_array_equal(a.input_durations, b.input_durations)

  # Sort species probabilities by species IDs before comparison
  sort_idx_a = np.argsort(a.species_ids, axis=-1)
  sort_idx_b = np.argsort(b.species_ids, axis=-1)

  sorted_ids_a = np.take_along_axis(a.species_ids, sort_idx_a, axis=-1)
  sorted_ids_b = np.take_along_axis(b.species_ids, sort_idx_b, axis=-1)

  # order may differ due to different top-k selection, but ids must be the same
  np.testing.assert_array_equal(sorted_ids_a, sorted_ids_b)

  sorted_probs_a = np.take_along_axis(a.species_probs, sort_idx_a, axis=-1)
  sorted_probs_b = np.take_along_axis(b.species_probs, sort_idx_b, axis=-1)
  np.testing.assert_array_equal(sorted_probs_a, sorted_probs_b)

  sorted_masks_a = np.take_along_axis(a.species_masked, sort_idx_a, axis=-1)
  sorted_masks_b = np.take_along_axis(b.species_masked, sort_idx_b, axis=-1)
  np.testing.assert_array_equal(sorted_masks_a, sorted_masks_b)


def assert_encoding_result_is_close(
  a: AcousticEncodingResultBase, b: AcousticEncodingResultBase, max_abs_diff: float
) -> None:
  assert isinstance(a, AcousticEncodingResultBase)
  assert isinstance(b, AcousticEncodingResultBase)

  np.testing.assert_array_equal(a.segment_duration_s, b.segment_duration_s)
  np.testing.assert_array_equal(a.overlap_duration_s, b.overlap_duration_s)
  np.testing.assert_array_equal(a.inputs, b.inputs)
  np.testing.assert_array_equal(a.input_durations, b.input_durations)

  np.testing.assert_array_equal(a.embeddings.shape, b.embeddings.shape)
  max_abs = get_max_absolute_tolerance(
    a.embeddings,
    b.embeddings,
  )

  assert max_abs <= max_abs_diff, (
    f"Max absolute difference {max_abs} exceeds threshold {max_abs_diff}"
  )


def assert_prediction_result_is_close(
  a: AcousticPredictionResultBase, b: AcousticPredictionResultBase, max_abs_diff: float
) -> None:
  assert isinstance(a, AcousticPredictionResultBase)
  assert isinstance(b, AcousticPredictionResultBase)

  np.testing.assert_array_equal(a.segment_duration_s, b.segment_duration_s)
  np.testing.assert_array_equal(a.overlap_duration_s, b.overlap_duration_s)
  np.testing.assert_array_equal(a.inputs, b.inputs)
  np.testing.assert_array_equal(a.species_list, b.species_list)
  np.testing.assert_array_equal(a.input_durations, b.input_durations)

  # Sort species probabilities by species IDs before comparison
  sort_idx_a = np.argsort(a.species_ids, axis=-1)
  sort_idx_b = np.argsort(b.species_ids, axis=-1)

  sorted_ids_a = np.take_along_axis(a.species_ids, sort_idx_a, axis=-1)
  sorted_ids_b = np.take_along_axis(b.species_ids, sort_idx_b, axis=-1)

  # order may differ due to different top-k selection, but ids must be the same
  np.testing.assert_array_equal(sorted_ids_a, sorted_ids_b)

  sorted_masks_a = np.take_along_axis(a.species_masked, sort_idx_a, axis=-1)
  sorted_masks_b = np.take_along_axis(b.species_masked, sort_idx_b, axis=-1)
  np.testing.assert_array_equal(sorted_masks_a, sorted_masks_b)

  sorted_probs_a = np.take_along_axis(a.species_probs, sort_idx_a, axis=-1)
  sorted_probs_b = np.take_along_axis(b.species_probs, sort_idx_b, axis=-1)

  np.testing.assert_array_equal(sorted_probs_a.shape, sorted_probs_b.shape)
  max_abs = get_max_absolute_tolerance(
    sorted_probs_a,
    sorted_probs_b,
  )

  assert max_abs <= max_abs_diff, (
    f"Max absolute difference {max_abs} exceeds threshold {max_abs_diff}"
  )


def _check_tf_gpu() -> bool:
  try:
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    return len(devices) > 0
  except Exception:
    return False


def tensorflow_gpu_available() -> bool:
  ctx = mp.get_context()
  with ctx.Pool(1) as pool:
    result = pool.apply(_check_tf_gpu)
  return result


def ensure_gpu_or_skip() -> None:
  if not tensorflow_gpu_available():
    pytest.skip("GPU not available")


def ensure_gpu_or_skip_smi() -> None:
  gpu_available = False
  try:
    subprocess.check_output("nvidia-smi")
    gpu_available = True
  except Exception:
    pass
  if not gpu_available:
    pytest.skip("Nvidia GPU not available")


def ensure_gpu_or_skip_old() -> None:
  cuda_available = importlib.util.find_spec("nvidia", "cuda_runtime") is not None
  if not cuda_available:
    pytest.skip("Nvidia CUDA runtime not available")


def ensure_litert_or_skip() -> None:
  if not litert_installed():
    pytest.skip("litert library is not available")


def use_forkserver_or_skip() -> None:
  if "forkserver" in get_all_start_methods():
    set_start_method("forkserver", force=True)
  else:
    pytest.skip("forkserver start method not available on this platform")


def use_fork_or_skip() -> None:
  if "fork" in get_all_start_methods():
    set_start_method("fork", force=True)
  else:
    pytest.skip("fork start method not available on this platform")


def use_spawn_or_skip() -> None:
  if "spawn" in get_all_start_methods():
    set_start_method("spawn", force=True)
  else:
    pytest.skip("spawn start method not available on this platform")


@contextlib.contextmanager
def memory_monitor() -> Generator[Callable, None, None]:
  """Context manager für Memory-Monitoring."""
  process = psutil.Process(os.getpid())
  memory_before = process.memory_full_info().uss
  max_memory = ctypes.c_float(memory_before)
  stop_event = threading.Event()

  def monitor_worker() -> None:
    while not stop_event.is_set():
      try:
        current_memory = process.memory_full_info().uss
        if current_memory > max_memory.value:
          max_memory.value = current_memory
        time.sleep(0.05)
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        break

  monitor_thread = threading.Thread(
    target=monitor_worker, daemon=True, name="MemoryMonitorThread"
  )
  monitor_thread.start()

  def get_memory_delta() -> float:
    return (max_memory.value - memory_before) / 1024**2

  try:
    yield get_memory_delta
  finally:
    stop_event.set()
    monitor_thread.join(timeout=1.0)


@contextlib.contextmanager
def duration_counter() -> Generator[Callable, None, None]:
  """Context manager to measure duration of a code block."""
  start = time.perf_counter()

  def get_duration() -> float:
    end = time.perf_counter()
    return end - start

  try:
    yield get_duration
  finally:
    pass  # No cleanup needed


def get_max_absolute_tolerance(a: np.ndarray, b: np.ndarray):
  a = np.asarray(a)
  b = np.asarray(b)

  diff = np.abs(a - b)

  mean = np.mean(diff)
  std = np.std(diff)
  minimum = np.min(diff)
  maximum = np.max(diff)
  return maximum


def get_max_absolute_tolerance_threshold(
  a: np.ndarray, b: np.ndarray, threshold: float = 0.1
) -> float:
  a = np.asarray(a)
  b = np.asarray(b)

  mask = a > threshold
  if not np.any(mask):
    return 0.0

  diff = np.abs(a - b)
  masked_diff = diff[mask]

  mean = np.mean(masked_diff)
  std = np.std(masked_diff)
  minimum = np.min(masked_diff)
  maximum = np.max(masked_diff)
  return mean


def estimate_best_rtol_atol(a, b):
  a = np.asarray(a)
  b = np.asarray(b)

  diff = np.abs(a - b)
  denom = np.abs(b)

  with np.errstate(divide="ignore", invalid="ignore"):
    rel = np.where(denom == 0, np.inf, diff / denom)

  max_rel = np.max(rel)
  max_abs = np.max(diff)

  return max_rel, max_abs


def get_max_relative_tolerance(a: np.ndarray, b: np.ndarray):
  a = np.asarray(a)
  b = np.asarray(b)

  diff = np.abs(a - b)
  denom = np.abs(b)

  # Sonderfall: b == 0 → relative Fehler ist nur sinnvoll, wenn diff ebenfalls 0 ist
  with np.errstate(divide="ignore", invalid="ignore"):
    rel = np.where(denom == 0, np.inf, diff / denom)

  return np.max(rel)


def worst_decimal_precision(a: np.ndarray, b: np.ndarray, max_decimals: int = 8):
  diff = np.abs(a - b)
  max_diff = np.max(diff)

  for decimals in range(max_decimals, 0, -1):
    if max_diff < 10 ** (-decimals):
      return decimals
  return 0
