from pathlib import Path

import pytest

from birdnet.geo.models.v3_0.pb import _PB_DL_URL, GeoPBDownloaderV3_0
from birdnet.utils.helper import write_source_marker
from birdnet_tests.helper import create_fake_saved_model


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  model_dir = tmp_path / "model-fp32"
  create_fake_saved_model(model_dir)
  monkeypatch.setattr(
    GeoPBDownloaderV3_0, "_get_model_path", classmethod(lambda cls: model_dir)
  )
  monkeypatch.setattr(
    GeoPBDownloaderV3_0, "_check_labels_available", classmethod(lambda cls: True)
  )
  return model_dir


def test_cached_model_without_marker_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _prepare(tmp_path, monkeypatch)

  assert not GeoPBDownloaderV3_0._check_geo_model_available()


def test_cached_model_from_another_release_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, "https://example.org/geomodel/v3.0.3/older.zip")

  assert not GeoPBDownloaderV3_0._check_geo_model_available()


def test_cached_model_from_the_current_release_is_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, _PB_DL_URL)

  assert GeoPBDownloaderV3_0._check_geo_model_available()
