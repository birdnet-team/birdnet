"""A cold cache: one download, the members in place, the marker written."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.geo.models.v2_4.pb import _PB_DL_URL, GeoPBDownloaderV2_4
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.downloads import check_source_marker
from birdnet_tests.helper import fake_zip_downloads, label_members, saved_model_members

pytestmark = pytest.mark.no_tf


def test_cold_cache_downloads_once_and_marks_the_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  archive = saved_model_members("meta-model/") | label_members(
    "labels/", GeoPBDownloaderV2_4.AVAILABLE_LANGUAGES
  )
  calls = fake_zip_downloads(
    monkeypatch, {_PB_DL_URL: archive}, held_lock=GeoPBDownloaderV2_4._lock_dir()
  )

  model_dir, labels = GeoPBDownloaderV2_4.get_model_path_and_labels("en_us")
  GeoPBDownloaderV2_4.get_model_path_and_labels("de")

  assert calls == [_PB_DL_URL]
  assert model_dir == GeoPBDownloaderV2_4._get_paths()[0]
  assert check_source_marker(model_dir, _PB_DL_URL)
  assert list(labels) == ["Species_species_Species"]
  assert not GeoPBDownloaderV2_4._lock_dir().exists()
