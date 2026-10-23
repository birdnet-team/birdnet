"""A cold cache: one download, the model in place, labels regenerated afterwards."""

from __future__ import annotations

from pathlib import Path

import pytest

import birdnet.acoustic.models.v3_0.pb as pb_module
from birdnet.acoustic.models.v3_0.pb import _PB_DL_URL, AcousticPBDownloaderV3_0
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.helper import check_source_marker, write_source_marker
from birdnet_tests.helper import (
  create_fake_lang_dir,
  fake_zip_downloads,
  saved_model_members,
)

pytestmark = pytest.mark.no_tf


@pytest.fixture
def generated_labels(monkeypatch: pytest.MonkeyPatch) -> None:
  """Stand in for the label generation, which needs its own downloads."""
  monkeypatch.setattr(
    AcousticPBDownloaderV3_0,
    "ensure_labels_available",
    classmethod(
      lambda cls: create_fake_lang_dir(cls._get_paths()[1], cls.AVAILABLE_LANGUAGES)
    ),
  )


def test_cold_cache_downloads_once_and_marks_the_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated_labels: None
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  calls = fake_zip_downloads(
    monkeypatch,
    {_PB_DL_URL: saved_model_members("")},
    held_lock=AcousticPBDownloaderV3_0._lock_dir(),
  )

  model_dir, labels = AcousticPBDownloaderV3_0.get_model_path_and_labels("en_us")
  AcousticPBDownloaderV3_0.get_model_path_and_labels("de")

  assert calls == [_PB_DL_URL]
  assert model_dir == AcousticPBDownloaderV3_0._get_paths()[0]
  assert check_source_marker(model_dir, _PB_DL_URL)
  assert list(labels) == ["Species_species_Species"]
  assert not AcousticPBDownloaderV3_0._lock_dir().exists()


def test_stale_labels_are_gone_before_the_model_reads_as_current(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated_labels: None
) -> None:
  """Once the marker is in place another process may generate labels; a wipe
  after that point would take them from under it."""
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  fake_zip_downloads(monkeypatch, {_PB_DL_URL: saved_model_members("")})
  _, lang_dir = AcousticPBDownloaderV3_0._get_paths()
  create_fake_lang_dir(lang_dir, AcousticPBDownloaderV3_0.AVAILABLE_LANGUAGES)
  labels_present_at_marker: list[bool] = []

  def marker_then_record(model_dir: Path, dl_url: str) -> None:
    labels_present_at_marker.append(lang_dir.exists())
    write_source_marker(model_dir, dl_url)

  monkeypatch.setattr(pb_module, "write_source_marker", marker_then_record)

  AcousticPBDownloaderV3_0.get_model_path_and_labels("en_us")

  assert labels_present_at_marker == [False]
