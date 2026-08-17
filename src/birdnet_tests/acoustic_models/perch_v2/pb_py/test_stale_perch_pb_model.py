from pathlib import Path

import pytest

from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
from birdnet.utils.helper import write_source_marker
from birdnet_tests.helper import create_fake_saved_model


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  model_dir = tmp_path / "perch-v2-cpu"
  labels_path = model_dir / "assets" / "labels.csv"
  create_fake_saved_model(model_dir)
  labels_path.parent.mkdir(parents=True, exist_ok=True)
  labels_path.write_text("species\n", encoding="utf-8")
  monkeypatch.setattr(
    AcousticPBDownloaderPerchV2,
    "_get_paths",
    classmethod(lambda cls, device: (model_dir, labels_path)),
  )
  return model_dir


def test_cached_model_without_marker_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _prepare(tmp_path, monkeypatch)

  assert not AcousticPBDownloaderPerchV2._check_acoustic_model_available("CPU")


def test_cached_model_from_another_upload_is_not_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, "https://tuc.cloud/index.php/s/older/download")

  assert not AcousticPBDownloaderPerchV2._check_acoustic_model_available("CPU")


def test_cached_model_from_the_current_upload_is_available(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, AcousticPBDownloaderPerchV2.MODEL_DOWNLOAD_URL_CPU)

  assert AcousticPBDownloaderPerchV2._check_acoustic_model_available("CPU")


def test_gpu_marker_does_not_satisfy_the_cpu_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_dir = _prepare(tmp_path, monkeypatch)
  write_source_marker(model_dir, AcousticPBDownloaderPerchV2.MODEL_DOWNLOAD_URL_GPU)

  assert not AcousticPBDownloaderPerchV2._check_acoustic_model_available("CPU")
