"""The premise of starting children with one OpenBLAS thread, where it pays.

OpenBLAS commits a buffer for every thread of its pool as it loads, and Windows
counts that commit against its limit whether the pages are touched or not.
"""

from __future__ import annotations

import os
import subprocess
import sys

import psutil
import pytest

pytestmark = pytest.mark.no_tf

_MIB = 2**20
# OpenBLAS reads the first of these that is set.
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
_PROBE = "import numpy, psutil; print(psutil.Process().memory_info().private)"
# Each thread after the first commits 32 MiB; four logical cores give three.
_MIN_CORES = 4
_MIN_SAVING_BYTES = 64 * _MIB


def _numpy_uses_openblas() -> bool:
  import numpy as np

  try:
    config = np.show_config(mode="dicts")
  except TypeError:  # an older numpy without the dict mode
    return False
  blas = config.get("Build Dependencies", {}).get("blas", {})
  return "openblas" in str(blas.get("name", "")).lower()


def _commit_after_numpy_import(openblas_threads: str | None) -> int:
  env = {k: v for k, v in os.environ.items() if k.upper() not in _THREAD_VARS}
  if openblas_threads is not None:
    env["OPENBLAS_NUM_THREADS"] = openblas_threads
  out = subprocess.run(
    [sys.executable, "-c", _PROBE],
    env=env,
    capture_output=True,
    text=True,
    check=True,
    timeout=120,
  )
  return int(out.stdout.strip())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' commit accounting")
def test_one_openblas_thread_commits_less() -> None:
  if (psutil.cpu_count() or 1) < _MIN_CORES:
    pytest.skip(f"fewer than {_MIN_CORES} logical cores")
  if not _numpy_uses_openblas():
    pytest.skip("numpy is not built with OpenBLAS")
  uncapped = _commit_after_numpy_import(None)
  capped = _commit_after_numpy_import("1")
  assert uncapped - capped >= _MIN_SAVING_BYTES, (
    f"commit {uncapped / _MIB:.0f} MiB uncapped, {capped / _MIB:.0f} MiB with "
    f"one thread"
  )
