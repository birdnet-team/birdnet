from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import birdnet.acoustic.inference.resources as resources_module
from birdnet.acoustic.inference.resources import StatisticsResources
from birdnet.core.base import get_session_id_hash
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet_tests.fake_acoustic_backend import colliding_session_ids

pytestmark = pytest.mark.no_tf


class _OneSecond(datetime):
  @classmethod
  def now(cls, tz: object = None) -> "_OneSecond":
    return cls(2026, 1, 2, 3, 4, 5)


def _benchmark_session_dir(session_id: str) -> Path:
  config = SimpleNamespace(
    start_method="spawn",
    output_conf=SimpleNamespace(show_stats="benchmark", progress_callback=None),
  )
  resources = StatisticsResources.create(
    session_id=session_id,
    conf=config,  # type: ignore[arg-type]
    benchmark_dir_name="probe",
  )
  assert resources.benchmark_session_dir is not None
  return resources.benchmark_session_dir


def test_sessions_whose_short_hashes_collide_get_their_own_benchmark_directory(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # The directory holds a session's log and run outputs. Its name starts with
  # the second the session began in, which two sessions can share.
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  monkeypatch.setattr(resources_module, "datetime", _OneSecond)
  first, second = colliding_session_ids()
  assert get_session_id_hash(first) == get_session_id_hash(second)

  directories = [_benchmark_session_dir(first), _benchmark_session_dir(second)]

  assert directories[0] != directories[1]
  assert all(directory.is_dir() for directory in directories)
  assert all(tmp_path in directory.parents for directory in directories)
