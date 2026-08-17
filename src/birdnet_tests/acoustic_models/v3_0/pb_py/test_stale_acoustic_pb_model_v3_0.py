from pathlib import Path

import pytest

from birdnet.acoustic.models.v3_0.pb import _PB_DL_URL, AcousticPBDownloaderV3_0
from birdnet.utils.helper import write_source_marker
from birdnet_tests.helper import create_fake_saved_model


def test_cached_model_without_marker_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = tmp_path / "model"
  create_fake_saved_model(model_dir)
  monkeypatch.setattr(
    AcousticPBDownloaderV3_0,
    "_get_paths",
    classmethod(lambda cls: (model_dir, tmp_path / "lang")),
  )

  assert not AcousticPBDownloaderV3_0._check_model_files_available()


def test_cached_model_from_another_release_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = tmp_path / "model"
  create_fake_saved_model(model_dir)
  write_source_marker(model_dir, "https://zenodo.org/records/1/files/older.zip")
  monkeypatch.setattr(
    AcousticPBDownloaderV3_0,
    "_get_paths",
    classmethod(lambda cls: (model_dir, tmp_path / "lang")),
  )

  assert not AcousticPBDownloaderV3_0._check_model_files_available()


def test_cached_model_from_the_current_release_is_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = tmp_path / "model"
  create_fake_saved_model(model_dir)
  write_source_marker(model_dir, _PB_DL_URL)
  monkeypatch.setattr(
    AcousticPBDownloaderV3_0,
    "_get_paths",
    classmethod(lambda cls: (model_dir, tmp_path / "lang")),
  )

  assert AcousticPBDownloaderV3_0._check_model_files_available()
