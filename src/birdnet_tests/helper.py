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

import psutil
import pytest

from birdnet.backends import litert_installed


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

  monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
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
