"""A real worker of a real session sees one OpenBLAS thread, end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.core.child_env import OPENBLAS_THREADS_VAR
from birdnet_tests.fake_acoustic_backend import (
  OneBlasThreadBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_N_SEGMENTS = 4


def test_the_worker_starts_with_one_blas_thread(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  wav = write_marked_audio(tmp_path / "in.wav", _N_SEGMENTS)
  with fake_predict_session(
    tmp_path, n_workers=1, backend=OneBlasThreadBackend
  ) as session:
    result = session.run(str(wav))
  assert result.max_n_segments == _N_SEGMENTS
  assert len(result.to_structured_array()) > 0
