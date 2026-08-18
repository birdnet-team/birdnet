# birdnet/logging_utils.py
from __future__ import annotations

import logging
import os
import sys
import tempfile
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
  on the package logger at DEBUG, which an embedding application can record by
  attaching a handler — the default configuration attaches none, so warnings
  that never raise are seen by re-running with `BIRDNET_TF_VERBOSE=1`.
  """
  if native_output_is_verbose():
    yield
    return

  try:
    saved_stderr_fd = os.dup(2)
  except OSError:
    # No usable stderr, e.g. under pythonw.exe. Nothing to suppress.
    yield
    return

  try:
    with tempfile.TemporaryFile() as capture:
      sys.stderr.flush()
      os.dup2(capture.fileno(), 2)
      failed = False
      try:
        yield
      except BaseException:
        failed = True
        raise
      finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, 2)
        capture.seek(0)
        captured = capture.read().decode("utf-8", "replace")
        if captured:
          if failed:
            sys.stderr.write(captured)
            sys.stderr.flush()
          else:
            # Warnings that do not raise — a CUDA library that could not be
            # loaded, say — explain later behaviour and must not be destroyed.
            get_logger_for_package(__name__).debug(
              "Suppressed native output:\n%s", captured.rstrip()
            )
  finally:
    os.close(saved_stderr_fd)
