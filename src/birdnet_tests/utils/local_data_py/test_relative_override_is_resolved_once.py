"""A relative ``BIRDNET_APP_DATA`` keeps the directory it was first resolved against."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.local_data import get_model_root_dir

pytestmark = pytest.mark.no_tf


def test_a_relative_override_does_not_follow_a_later_chdir(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  first, second = tmp_path / "first", tmp_path / "second"
  first.mkdir()
  second.mkdir()
  monkeypatch.setenv(ENV_VAR_APP_DATA, f"cache-{uuid.uuid4().hex}")
  monkeypatch.chdir(first)
  before = get_model_root_dir("geo", "2.4", "tf")

  monkeypatch.chdir(second)
  after = get_model_root_dir("geo", "2.4", "tf")

  assert before == after
  assert before.is_relative_to(first.resolve())
