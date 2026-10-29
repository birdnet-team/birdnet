"""An exception inside the model names itself in the caller's error and the log."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import (
  RaisingBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf


def test_the_models_exception_text_reaches_the_caller_and_the_log(
  tmp_path: Path,
) -> None:
  files = [str(write_marked_audio(tmp_path / f"in_{i}.wav", 8)) for i in range(2)]
  with fake_predict_session(tmp_path, n_workers=2, backend=RaisingBackend) as session:
    log_file = session._resources.logging_resources.session_log_file
    with pytest.raises(RuntimeError) as info:
      session.run(files)
  text = str(info.value)
  assert "synthetic model failure" in text, text
  assert "Worker" in text, text
  assert "Analysis was cancelled" in text, text
  log = log_file.read_text(errors="replace")
  assert "synthetic model failure" in log
  assert "Traceback" in log


def test_the_error_is_in_the_log_at_the_default_level(tmp_path: Path) -> None:
  files = [str(write_marked_audio(tmp_path / f"in_{i}.wav", 8)) for i in range(2)]
  package = logging.getLogger("birdnet")
  level = package.level
  package.setLevel(logging.INFO)
  try:
    with fake_predict_session(tmp_path, n_workers=1, backend=RaisingBackend) as session:
      log_file = session._resources.logging_resources.session_log_file
      with pytest.raises(RuntimeError):
        session.run(files)
  finally:
    package.setLevel(level)
  assert "synthetic model failure" in log_file.read_text(errors="replace")
