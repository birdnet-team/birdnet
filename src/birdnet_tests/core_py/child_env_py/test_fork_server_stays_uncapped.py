"""A fork server that birdnet's start launches must not keep a BLAS cap.

The server outlives that start and forks every later child of the application,
so a variable it inherited would reach all of them. A fresh interpreter, so
that birdnet's start is the one that launches the server.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys

import pytest

from birdnet.core.child_env import OPENBLAS_THREADS_VAR
from birdnet.globals import ENV_VAR_START_METHOD

pytestmark = pytest.mark.no_tf

_SCRIPT = f"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

from birdnet.core.backends import run_in_subprocess

if __name__ == "__main__":
  mp.set_start_method("forkserver")
  run_in_subprocess(os.getpid, (), timeout_s=60)
  with ProcessPoolExecutor(1, mp_context=mp.get_context("forkserver")) as pool:
    print(repr(pool.submit(os.getenv, "{OPENBLAS_THREADS_VAR}").result()))
"""


@pytest.mark.skipif(
  "forkserver" not in mp.get_all_start_methods(), reason="no forkserver here"
)
def test_the_applications_later_children_get_no_cap() -> None:
  env = {
    k: v
    for k, v in os.environ.items()
    if k not in (OPENBLAS_THREADS_VAR, ENV_VAR_START_METHOD)
  }
  out = subprocess.run(
    [sys.executable, "-c", _SCRIPT],
    env=env,
    capture_output=True,
    text=True,
    timeout=180,
    check=False,
  )
  assert out.returncode == 0, out.stderr
  assert out.stdout.strip() == "None", out.stdout
