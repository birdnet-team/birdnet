"""The environment birdnet's own child processes start with."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

OPENBLAS_THREADS_VAR = "OPENBLAS_NUM_THREADS"

_lock = threading.Lock()
_n_inside = 0
_set_here = False


@contextmanager
def one_blas_thread_in_children(start_method: str) -> Iterator[None]:
  """Start the processes created in this block with one OpenBLAS thread.

  numpy and scipy each load their own OpenBLAS, which sizes its thread pool to
  the logical cores and commits about 32 MiB per thread as it loads. No child
  makes a BLAS call on its hot path, so the memory is never touched, but
  Windows counts it against the commit limit. Under ``spawn`` multiprocessing's
  bootstrap imports numpy in the child before any birdnet code runs there, so
  the limit has to be in the environment the child is created with: the
  parent's ``os.environ`` while ``start()`` runs. A value the user has set is
  left alone, and overlapping blocks keep the variable until the last one ends.

  Other start methods are left alone: under ``fork`` the child inherits the
  parent's loaded OpenBLAS, so the variable would change nothing, and under
  ``forkserver`` a server this start launches would keep it and pass it to
  every later child of the application.
  """
  if start_method != "spawn":
    yield
    return
  global _n_inside, _set_here
  with _lock:
    if _n_inside == 0:
      _set_here = OPENBLAS_THREADS_VAR not in os.environ
      if _set_here:
        # Processes the application starts from its own threads meanwhile
        # inherit it too, and an OpenBLAS this process first loads meanwhile
        # keeps one thread.
        os.environ[OPENBLAS_THREADS_VAR] = "1"
    _n_inside += 1
  try:
    yield
  finally:
    with _lock:
      _n_inside -= 1
      if _n_inside == 0 and _set_here:
        _set_here = False
        # A value the application set meanwhile is its own.
        if os.environ.get(OPENBLAS_THREADS_VAR) == "1":
          del os.environ[OPENBLAS_THREADS_VAR]
