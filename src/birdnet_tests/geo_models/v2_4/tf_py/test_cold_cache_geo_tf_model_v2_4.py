"""A cold cache: one download, the model file and the labels in place."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.geo.models.v2_4.tf import GeoTFDownloaderV2_4
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet_tests.helper import fake_zip_downloads, label_members

pytestmark = pytest.mark.no_tf


def test_cold_cache_downloads_once(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  info = GeoTFDownloaderV2_4._model_info
  archive = {info.dl_file_name: b"\0" * info.file_size} | label_members(
    "labels/", GeoTFDownloaderV2_4.AVAILABLE_LANGUAGES
  )
  calls = fake_zip_downloads(
    monkeypatch, {info.dl_url: archive}, held_lock=GeoTFDownloaderV2_4._lock_dir()
  )

  model_path, labels = GeoTFDownloaderV2_4.get_model_path_and_labels("en_us")
  GeoTFDownloaderV2_4.get_model_path_and_labels("de")

  assert calls == [info.dl_url]
  assert model_path == GeoTFDownloaderV2_4._get_paths()[0]
  assert model_path.stat().st_size == info.file_size
  assert list(labels) == ["Species_species_Species"]
  assert not GeoTFDownloaderV2_4._lock_dir().exists()
