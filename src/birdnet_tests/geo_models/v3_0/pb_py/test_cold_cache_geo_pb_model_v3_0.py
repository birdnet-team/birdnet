"""A cold cache: one download, the model in place, the marker written."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.geo.models.v3_0.pb import _PB_DL_URL, GeoPBDownloaderV3_0
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.helper import check_source_marker
from birdnet_tests.helper import (
  create_fake_lang_dir,
  fake_zip_downloads,
  saved_model_members,
)

pytestmark = pytest.mark.no_tf


def test_cold_cache_downloads_once_and_marks_the_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  # The label generation needs its own downloads; stand in for it.
  monkeypatch.setattr(
    GeoPBDownloaderV3_0,
    "ensure_labels_available",
    classmethod(
      lambda cls: create_fake_lang_dir(cls._get_lang_dir(), cls.AVAILABLE_LANGUAGES)
    ),
  )
  monkeypatch.setattr(
    GeoPBDownloaderV3_0, "_check_labels_available", classmethod(lambda cls: True)
  )
  archive = saved_model_members("BirdNET+_Geomodel_V3.0.4_Global_14K_FP32_TF/")
  calls = fake_zip_downloads(
    monkeypatch, {_PB_DL_URL: archive}, held_lock=GeoPBDownloaderV3_0._lock_dir()
  )

  model_dir, labels = GeoPBDownloaderV3_0.get_model_path_and_labels("en_us")
  GeoPBDownloaderV3_0.get_model_path_and_labels("de")

  assert calls == [_PB_DL_URL]
  assert model_dir == GeoPBDownloaderV3_0._get_model_path()
  assert check_source_marker(model_dir, _PB_DL_URL)
  assert list(labels) == ["Species_species_Species"]
  assert not GeoPBDownloaderV3_0._lock_dir().exists()
