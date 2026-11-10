"""Two sessions alive at once keep their shared memory and their log apart.

Both are named after a hash of the session id, long enough that two live
sessions never share it. With shared names the second session fails to start,
on every platform: a segment is created exclusively and never removed first,
which RingBufferResources/test_create.py pins.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

import birdnet.core.base as base_module
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.session import AcousticEncodingSession
from birdnet_tests.fake_acoustic_backend import (
  colliding_session_ids,
  fake_encode_session,
)

pytestmark = pytest.mark.no_tf

_N_SEGMENTS = 60
_SAMPLES_PER_SEGMENT = 144_000


def _constant_audio(value: float) -> tuple[np.ndarray, int]:
  samples = np.full(_N_SEGMENTS * _SAMPLES_PER_SEGMENT, value, dtype=np.float32)
  return samples, 48_000


def test_sessions_whose_short_hashes_collide_run_at_once_on_their_own_data(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  ids = iter(colliding_session_ids())
  monkeypatch.setattr(base_module, "get_session_id", lambda: next(ids))
  sessions = {
    value: fake_encode_session(tmp_path, n_workers=2, n_producers=2, batch_size=4)
    for value in (0.25, 0.75)
  }
  outcomes: dict[float, AcousticEncodingResultBase | BaseException] = {}

  def run(value: float, session: AcousticEncodingSession) -> None:
    try:
      outcomes[value] = session.run_arrays(_constant_audio(value))
    except BaseException as error:  # noqa: BLE001
      outcomes[value] = error

  first, second = sessions.values()
  with first, second:
    logs = [s._resources.logging_resources.session_log_file for s in (first, second)]
    threads = [threading.Thread(target=run, args=item) for item in sessions.items()]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

  assert logs[0] != logs[1]
  for value, outcome in outcomes.items():
    assert isinstance(outcome, AcousticEncodingResultBase), outcome
    # The fake backend hands a segment's first sample back as the first
    # element of its embedding.
    np.testing.assert_array_equal(
      outcome.embeddings[0, :, 0], np.full(_N_SEGMENTS, value, dtype=np.float32)
    )
