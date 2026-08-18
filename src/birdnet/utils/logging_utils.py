# birdnet/logging_utils.py
from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from collections.abc import Generator
from contextlib import contextmanager

from birdnet.globals import ENV_VAR_TF_VERBOSE, PKG_NAME

"""
loggers:

root:
- birdnet (INFO)
  - birdnet.session_XXX (INFO, inherited)
    - logger for each predict/encode session
    - birdnet.session_XXX.modules... e.g. birdnet.session_XXX.birdnet.acoustic_models.inference_pipeline.processes
- birdnet_file_writer.session_XXX (INFO, inherited)
  - file writer for each predict/encode session

"""  # noqa: E501


def get_package_logger() -> logging.Logger:
  return logging.getLogger(PKG_NAME)


def get_logger_for_package(name: str) -> logging.Logger:
  logger = logging.getLogger(name)
  logger.parent = get_package_logger()
  return logger


def get_package_logging_level() -> int:
  result = get_package_logger().level
  return result


def init_package_logger(logging_level: int) -> None:
  root = get_package_logger()
  root.setLevel(logging_level)
  root.propagate = False


def native_output_is_verbose() -> bool:
  return os.environ.get(ENV_VAR_TF_VERBOSE, "0") not in ("", "0")


# File descriptor 2 is process-wide, so two threads redirecting it at once would
# restore each other's saved descriptor and leave the process without a usable
# stderr. Reentrant, because nested suppression unwinds in order on one thread.
_NATIVE_STDERR_LOCK = threading.RLock()


@contextmanager
def suppress_native_stderr() -> Generator[None, None, None]:
  """Hide what native code writes to stderr while the block runs.

  TensorFlow prints its absl banner and the oneDNN notice from C++ straight to
  file descriptor 2, before absl logging is initialized. `logging`, absl's
  verbosity and `TF_CPP_MIN_LOG_LEVEL` all act above that and cannot reach it;
  only redirecting the descriptor can. Every worker process imports TensorFlow,
  so the banner is printed once per process.

  If the block raises, the captured text is written to stderr, so the native
  diagnostics of a failed import still reach the user. Otherwise it is emitted
  on the package logger at DEBUG. No handler is attached by default, and worker
  processes have none at all, so in practice a warning that never raises is
  seen by re-running with `BIRDNET_TF_VERBOSE=1`.

  Suppression is a convenience, never a precondition: if stderr cannot be
  redirected the block still runs, unsuppressed. Because the descriptor is
  process-wide the block is serialized, which also means concurrent callers
  wait out an import that is already running.
  """
  stderr = sys.stderr
  if (
    native_output_is_verbose()
    # No usable stderr to redirect: pythonw.exe, a service, or `2>&-`. On
    # Windows os.dup(2) still succeeds there, so this has to be checked too.
    or stderr is None
    or not hasattr(stderr, "flush")
    or not hasattr(stderr, "write")
  ):
    yield
    return

  with _NATIVE_STDERR_LOCK:
    try:
      saved_stderr_fd = os.dup(2)
    except OSError:
      yield
      return

    try:
      try:
        # Not opened as a `with` here: creation has to be guarded on its own,
        # and the handle is closed by the `with capture` below.
        capture = tempfile.TemporaryFile()  # noqa: SIM115
      except OSError:
        # Read-only or full temp directory: run without suppressing rather
        # than turn a working import into a disk error.
        yield
        return

      with capture:
        stderr.flush()
        os.dup2(capture.fileno(), 2)
        failed = False
        try:
          yield
        except BaseException:
          failed = True
          raise
        finally:
          stderr.flush()
          os.dup2(saved_stderr_fd, 2)
          capture.seek(0)
          captured = capture.read().decode("utf-8", "replace")
          if captured:
            if failed:
              stderr.write(captured)
              stderr.flush()
            else:
              # Warnings that do not raise — a CUDA library that could not be
              # loaded, say — explain later behaviour and must not be destroyed.
              get_logger_for_package(__name__).debug(
                "Suppressed native output:\n%s", captured.rstrip()
              )
    finally:
      os.close(saved_stderr_fd)
