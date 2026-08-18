import io
import logging
import os
import sys
from pathlib import Path

import pytest

from birdnet.globals import ENV_VAR_TF_VERBOSE
from birdnet.utils.logging_utils import suppress_native_stderr


def _capture_fd2(target: Path):  # type: ignore[no-untyped-def]
  """Point fd 2 at a file, so writes bypassing sys.stderr can be inspected."""

  class _Redirect:
    def __enter__(self) -> None:
      self._saved = os.dup(2)
      self._fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
      os.dup2(self._fd, 2)

    def __exit__(self, *_: object) -> None:
      os.dup2(self._saved, 2)
      os.close(self._fd)
      os.close(self._saved)

  return _Redirect()


@pytest.mark.no_tf
def test_hides_native_writes_and_restores_afterwards(tmp_path: Path) -> None:
  out = tmp_path / "fd2.txt"

  with _capture_fd2(out):
    with suppress_native_stderr():
      os.write(2, b"NATIVE_NOISE\n")
    os.write(2, b"AFTER_BLOCK\n")

  written = out.read_text(encoding="utf-8")
  assert "NATIVE_NOISE" not in written
  assert "AFTER_BLOCK" in written


@pytest.mark.no_tf
def test_replays_captured_output_when_the_block_raises(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  replayed = io.StringIO()
  monkeypatch.setattr(sys, "stderr", replayed)

  with _capture_fd2(tmp_path / "fd2.txt"), pytest.raises(RuntimeError, match="boom"):
    with suppress_native_stderr():
      os.write(2, b"LOAD_DIAGNOSTIC\n")
      raise RuntimeError("boom")

  assert "LOAD_DIAGNOSTIC" in replayed.getvalue()


@pytest.mark.no_tf
def test_logs_suppressed_output_that_did_not_raise(tmp_path: Path) -> None:
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
    with _capture_fd2(tmp_path / "fd2.txt"):
      with suppress_native_stderr():
        os.write(2, b"Could not load dynamic library 'libcudart.so.12'\n")
  finally:
    logger.removeHandler(handler)
    logger.setLevel(previous_level)

  assert any("libcudart.so.12" in message for message in seen)


@pytest.mark.no_tf
def test_keeps_native_writes_when_verbose_is_requested(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_TF_VERBOSE, "1")
  out = tmp_path / "fd2.txt"

  with _capture_fd2(out):
    with suppress_native_stderr():
      os.write(2, b"NATIVE_NOISE\n")

  assert "NATIVE_NOISE" in out.read_text(encoding="utf-8")


@pytest.mark.no_tf
@pytest.mark.parametrize("value", ["", "0"])
def test_zero_and_empty_still_suppress(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
  monkeypatch.setenv(ENV_VAR_TF_VERBOSE, value)
  out = tmp_path / "fd2.txt"

  with _capture_fd2(out):
    with suppress_native_stderr():
      os.write(2, b"NATIVE_NOISE\n")

  assert "NATIVE_NOISE" not in out.read_text(encoding="utf-8")
