"""Entering one session object a second time logs the second run like the first."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import fake_predict_session

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000


def test_second_entry_logs_without_errors_and_detaches_again(
  tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
  samples = np.zeros(_SAMPLE_RATE * 3, dtype=np.float32)
  session = fake_predict_session(tmp_path, n_workers=1)
  session_logger = logging.getLogger(f"birdnet.session_{session._session_id}")
  writer_logger = logging.getLogger(
    f"birdnet_file_writer.session_{session._session_id}"
  )

  for _ in range(2):
    with session:
      session.run_arrays((samples, _SAMPLE_RATE))
      assert len(session_logger.handlers) == 1
      assert len(writer_logger.handlers) == 1
      assert session._resources is not None
      log_file = session._resources.logging_resources.session_log_file
    assert session_logger.handlers == []
    assert writer_logger.handlers == []
    assert log_file.stat().st_size > 0

  assert "Logging error" not in capfd.readouterr().err
