import contextlib
import ctypes
import os
import threading
import time
from collections.abc import Callable, Generator
from multiprocessing import get_all_start_methods, set_start_method

import psutil
import pytest


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
