import contextlib
import logging
import os
import sys
import threading

import pytest

from birdnet.globals import ENV_VAR_TF_VERBOSE
from birdnet.utils.logging_utils import suppress_native_stderr

# `capfd` reads file descriptors 1 and 2, which is where native code writes and
# the only level at which this can be observed. Redirecting fd 2 by hand instead
# fights pytest's own capture and depends on what ran before in the worker.


@pytest.mark.no_tf
def test_hides_native_writes_and_restores_afterwards(
  capfd: pytest.CaptureFixture,
) -> None:
  with suppress_native_stderr():
    os.write(2, b"NATIVE_NOISE\n")
  os.write(2, b"AFTER_BLOCK\n")

  err = capfd.readouterr().err
  assert "NATIVE_NOISE" not in err
  assert "AFTER_BLOCK" in err


@pytest.mark.no_tf
def test_replays_captured_output_when_the_block_raises(
  capfd: pytest.CaptureFixture,
) -> None:
  with pytest.raises(RuntimeError, match="boom"), suppress_native_stderr():
    os.write(2, b"LOAD_DIAGNOSTIC\n")
    raise RuntimeError("boom")

  assert "LOAD_DIAGNOSTIC" in capfd.readouterr().err


@pytest.mark.no_tf
def test_logs_suppressed_output_that_did_not_raise() -> None:
  """A warning that never raises still has to be recoverable from the log."""
  logger = logging.getLogger("birdnet.utils.logging_utils")
  seen: list[str] = []

  class _Collect(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
      seen.append(record.getMessage())

  handler = _Collect()
  previous_level = logger.level
  logger.addHandler(handler)
  logger.setLevel(logging.DEBUG)
  try:
    with suppress_native_stderr():
      os.write(2, b"Could not load dynamic library 'libcudart.so.12'\n")
  finally:
    logger.removeHandler(handler)
    logger.setLevel(previous_level)

  assert any("libcudart.so.12" in message for message in seen)


@pytest.mark.no_tf
def test_keeps_native_writes_when_verbose_is_requested(
  capfd: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_TF_VERBOSE, "1")

  with suppress_native_stderr():
    os.write(2, b"NATIVE_NOISE\n")

  assert "NATIVE_NOISE" in capfd.readouterr().err


@pytest.mark.no_tf
@pytest.mark.parametrize("value", ["", "0"])
def test_zero_and_empty_still_suppress(
  capfd: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
  monkeypatch.setenv(ENV_VAR_TF_VERBOSE, value)

  with suppress_native_stderr():
    os.write(2, b"NATIVE_NOISE\n")

  assert "NATIVE_NOISE" not in capfd.readouterr().err


@pytest.mark.no_tf
def test_runs_the_block_when_stderr_is_unusable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """pythonw.exe and services have `sys.stderr is None`; suppression is a
  convenience and must never stop the block from running."""
  monkeypatch.setattr(sys, "stderr", None)

  ran = False
  with suppress_native_stderr():
    ran = True

  assert ran


@pytest.mark.no_tf
def test_runs_the_block_when_the_capture_file_cannot_be_created(
  capfd: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A full or read-only temp directory must not turn a working import into an
  error; the block runs unsuppressed instead."""

  def _no_temp_file(*_: object, **__: object) -> None:
    raise OSError(28, "No space left on device")

  monkeypatch.setattr(
    "birdnet.utils.logging_utils.tempfile.TemporaryFile", _no_temp_file
  )

  with suppress_native_stderr():
    os.write(2, b"NOT_SUPPRESSED\n")

  assert "NOT_SUPPRESSED" in capfd.readouterr().err


@pytest.mark.no_tf
def test_leaves_stderr_usable_after_concurrent_use(
  capfd: pytest.CaptureFixture,
) -> None:
  """Two threads redirecting fd 2 must not restore each other's descriptor and
  leave the process mute for the rest of its life."""
  # Trips only if both threads are inside the block at once, which serialized
  # access makes impossible — then each waits out the timeout instead. Without
  # it the block is too short for the threads to overlap and nothing is proven.
  both_inside = threading.Barrier(2)

  def worker() -> None:
    with suppress_native_stderr():
      with contextlib.suppress(threading.BrokenBarrierError):
        both_inside.wait(timeout=0.5)
      os.write(2, b"INSIDE\n")

  threads = [threading.Thread(target=worker) for _ in range(2)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(timeout=30)
    assert not t.is_alive()

  os.write(2, b"STILL_ALIVE\n")
  err = capfd.readouterr().err
  assert "INSIDE" not in err
  assert "STILL_ALIVE" in err
