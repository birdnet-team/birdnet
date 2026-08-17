from pathlib import Path

import pytest

from birdnet.geo.models.v2_4.pb import _PB_DL_URL, GeoPBDownloaderV2_4
from birdnet.utils.helper import write_source_marker
from birdnet_tests.helper import create_fake_lang_dir, create_fake_saved_model


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  model_dir = tmp_path / "model"
  lang_dir = tmp_path / "lang"
  create_fake_saved_model(model_dir)
  create_fake_lang_dir(lang_dir, GeoPBDownloaderV2_4.AVAILABLE_LANGUAGES)
  monkeypatch.setattr(
    GeoPBDownloaderV2_4,
    "_get_paths",
    classmethod(lambda cls: (model_dir, lang_dir)),
  )
  return model_dir


def test_cached_model_without_marker_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _prepare(tmp_path, monkeypatch)

  assert not GeoPBDownloaderV2_4._check_geo_model_available()


def test_cached_model_from_another_release_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, "https://zenodo.org/records/1/files/older.zip")

  assert not GeoPBDownloaderV2_4._check_geo_model_available()


def test_cached_model_from_the_current_release_is_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, _PB_DL_URL)

  assert GeoPBDownloaderV2_4._check_geo_model_available()
