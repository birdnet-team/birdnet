import os
import sys
import time

import psutil
import pytest

from birdnet.core.backends import run_in_subprocess
from birdnet.core.child_env import OPENBLAS_THREADS_VAR
from birdnet_tests.fake_acoustic_backend import raise_foreign_load_error

pytestmark = pytest.mark.no_tf


def test_returns_what_the_function_returns_in_another_process() -> None:
  pid = run_in_subprocess(os.getpid, (), timeout_s=60)
  assert pid != os.getpid()


def test_the_child_starts_with_one_blas_thread(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  found = run_in_subprocess(os.getenv, (OPENBLAS_THREADS_VAR,), timeout_s=60)
  assert found == "1"
  assert OPENBLAS_THREADS_VAR not in os.environ


def test_raises_what_the_function_raises() -> None:
  with pytest.raises(ValueError, match=r"invalid literal"):
    run_in_subprocess(int, ("not a number",), timeout_s=60)


def test_a_non_builtin_exception_arrives_as_runtime_error_naming_it() -> None:
  """Unpickling the original would import its module into this process."""
  with pytest.raises(RuntimeError, match=r"^ForeignLoadError: bad weights$"):
    run_in_subprocess(raise_foreign_load_error, (), timeout_s=60)
  assert "birdnet_tests.foreign_load_error" not in sys.modules


def _workers() -> set[int]:
  """Our child processes, apart from multiprocessing's resource tracker, which
  POSIX starts with the first spawned process and keeps for the process's life."""
  workers: set[int] = set()
  for child in psutil.Process().children():
    try:
      cmdline = " ".join(child.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
      # Exited between the listing and this call: not a leftover.
      continue
    if "resource_tracker" not in cmdline:
      workers.add(child.pid)
  return workers


def test_a_hanging_function_is_killed_at_the_timeout() -> None:
  """Nothing may outlive the timeout: a worker left behind would hold up the
  interpreter's exit."""
  before = _workers()
  started = time.monotonic()
  with pytest.raises(TimeoutError, match=r"did not finish within 1 s"):
    run_in_subprocess(time.sleep, (60,), timeout_s=1)

  assert time.monotonic() - started < 10
  assert _workers() - before == set()


def test_a_function_that_cannot_be_pickled_starts_nothing() -> None:
  """On Windows a process whose start fails after creation stays alive and can
  answer the next call in its place."""
  before = _workers()
  with pytest.raises(Exception, match=r"[Pp]ickl"):
    run_in_subprocess(lambda: None, (), timeout_s=5)

  assert _workers() - before == set()
  assert run_in_subprocess(os.getpid, (), timeout_s=60) != os.getpid()
