"""Two benchmark sessions keep their folder, their log and their run outputs apart.

The folder is named after the second a session is entered in and a hash of its
id; the log and the outputs of every run are written inside it.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path

import pytest

import birdnet.acoustic.inference.resources as resources_module
import birdnet.core.base as base_module
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet_tests.fake_acoustic_backend import (
  colliding_session_ids,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf


class _OneSecond(datetime):
  @classmethod
  def now(cls, tz: object = None) -> _OneSecond:
    return cls(2026, 1, 2, 3, 4, 5)


def test_sessions_whose_short_hashes_collide_keep_their_benchmark_outputs_apart(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  app_data = tmp_path / "app"
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(app_data))
  ids = iter(colliding_session_ids())
  monkeypatch.setattr(base_module, "get_session_id", lambda: next(ids))
  audio = write_marked_audio(tmp_path / "audio.wav", 2)
  sessions = [
    fake_predict_session(tmp_path, n_workers=1, show_stats="benchmark")
    for _ in range(2)
  ]

  with contextlib.ExitStack() as live:
    with monkeypatch.context() as clock:
      clock.setattr(resources_module, "datetime", _OneSecond)
      for session in sessions:
        live.enter_context(session)
    folders = [s._resources.stats_resources.benchmark_session_dir for s in sessions]
    logs = [s._resources.logging_resources.session_log_file for s in sessions]
    for session in sessions:
      session.run(audio)

  assert folders[0] != folders[1]
  for folder, log in zip(folders, logs, strict=True):
    assert folder is not None
    assert app_data in folder.parents
    assert log.parent == folder
    assert log.is_file()
    assert len(list(folder.glob("run-1/*-stats.json"))) == 1
