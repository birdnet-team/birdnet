"""Leaving a session detaches the log handlers it attached in this process.

Session ids are unique and loggers live as long as the process, so a handler
left on one is never collected: every session would keep it, together with the
closed queue or log file it points at.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import fake_predict_session

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000


def test_exit_detaches_the_session_log_handlers(tmp_path: Path) -> None:
  samples = np.zeros(_SAMPLE_RATE * 3, dtype=np.float32)

  with fake_predict_session(tmp_path, n_workers=1) as session:
    session.run_arrays((samples, _SAMPLE_RATE))
    session_id = session._session_id
    session_logger = logging.getLogger(f"birdnet.session_{session_id}")
    writer_logger = logging.getLogger(f"birdnet_file_writer.session_{session_id}")
    # Attached while the session is open, so the checks below are not vacuous.
    assert session_logger.handlers
    assert writer_logger.handlers

  assert session_logger.handlers == []
  assert writer_logger.handlers == []
