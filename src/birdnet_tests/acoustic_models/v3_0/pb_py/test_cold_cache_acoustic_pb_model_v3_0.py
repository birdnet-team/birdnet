"""A cold cache: one download, the model in place, labels regenerated afterwards."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import birdnet.acoustic.models.v3_0.pb as pb_module
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.pb import _PB_DL_URL, AcousticPBDownloaderV3_0
from birdnet.globals import ENV_VAR_APP_DATA, MODEL_PRECISION_FP32
from birdnet.utils.downloads import check_source_marker, write_source_marker
from birdnet.utils.local_data import get_model_path
from birdnet_tests.helper import (
  create_fake_lang_dir,
  create_fake_saved_model,
  fake_labels,
  fake_zip_downloads,
  saved_model_members,
  zip_bytes,
)

pytestmark = pytest.mark.no_tf

_N_CLASSES = AcousticModelV3_0.get_n_classes()
# The released zip holds the SavedModel in this folder, not at its root.
_ARCHIVE = saved_model_members("BirdNET+_V3.0_Global_12K_FP32_Protobuf/")
_OTHER_RELEASE_URL = "https://zenodo.org/records/1/files/older.zip"


@pytest.fixture
def generated_labels(monkeypatch: pytest.MonkeyPatch) -> None:
  """Stand in for the label generation, which needs its own downloads."""
  monkeypatch.setattr(
    AcousticPBDownloaderV3_0,
    "ensure_labels_available",
    classmethod(
      lambda cls: create_fake_lang_dir(
        cls._get_paths()[1], cls.AVAILABLE_LANGUAGES, _N_CLASSES
      )
    ),
  )


@pytest.fixture
def pinned_to_the_served_zip(monkeypatch: pytest.MonkeyPatch) -> None:
  """Pin the digest of the zip these tests serve in place of the released one."""
  monkeypatch.setattr(
    pb_module, "_PB_DL_SHA256", hashlib.sha256(zip_bytes(_ARCHIVE)).hexdigest()
  )


def test_cold_cache_downloads_once_and_marks_the_model(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  generated_labels: None,
  pinned_to_the_served_zip: None,
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  calls = fake_zip_downloads(
    monkeypatch,
    {_PB_DL_URL: _ARCHIVE},
    held_lock=AcousticPBDownloaderV3_0._lock_dir(),
  )

  model_dir, labels = AcousticPBDownloaderV3_0.get_model_path_and_labels("en_us")
  AcousticPBDownloaderV3_0.get_model_path_and_labels("de")

  assert calls == [_PB_DL_URL]
  assert model_dir == AcousticPBDownloaderV3_0._get_paths()[0]
  assert check_source_marker(model_dir, _PB_DL_URL)
  assert list(labels) == fake_labels(_N_CLASSES)
  assert not AcousticPBDownloaderV3_0._lock_dir().exists()


def test_stale_labels_are_gone_before_the_model_reads_as_current(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  generated_labels: None,
  pinned_to_the_served_zip: None,
) -> None:
  """Once the marker is in place another process may generate labels; a wipe
  after that point would take them from under it."""
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  fake_zip_downloads(monkeypatch, {_PB_DL_URL: _ARCHIVE})
  _, lang_dir = AcousticPBDownloaderV3_0._get_paths()
  create_fake_lang_dir(
    lang_dir, AcousticPBDownloaderV3_0.AVAILABLE_LANGUAGES, _N_CLASSES
  )
  labels_present_at_marker: list[bool] = []

  def marker_then_record(model_dir: Path, dl_url: str) -> None:
    labels_present_at_marker.append(lang_dir.exists())
    write_source_marker(model_dir, dl_url)

  monkeypatch.setattr(pb_module, "write_source_marker", marker_then_record)

  AcousticPBDownloaderV3_0.get_model_path_and_labels("en_us")

  assert labels_present_at_marker == [False]


def test_a_zip_of_another_digest_is_not_unpacked(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated_labels: None
) -> None:
  """The pinned digest is the released zip's, which the served one cannot have."""
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  fake_zip_downloads(monkeypatch, {_PB_DL_URL: _ARCHIVE})

  with pytest.raises(RuntimeError, match="does not match its expected checksum"):
    AcousticPBDownloaderV3_0.get_model_path_and_labels("en_us")

  model_dir, _ = AcousticPBDownloaderV3_0._get_paths()
  assert list(model_dir.parent.iterdir()) == []


def test_a_model_directory_of_another_version_is_left_alone(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  generated_labels: None,
  pinned_to_the_served_zip: None,
) -> None:
  """Versions up to 1.4.0 keep their SavedModel in ``model-fp32``; two installs
  sharing an app data directory must not replace each other's."""
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  other = get_model_path("acoustic", "3.0", "pb", MODEL_PRECISION_FP32)
  create_fake_saved_model(other)
  (other / "saved_model.pb").write_bytes(b"another release")
  write_source_marker(other, _OTHER_RELEASE_URL)
  fake_zip_downloads(monkeypatch, {_PB_DL_URL: _ARCHIVE})

  model_dir, _ = AcousticPBDownloaderV3_0.get_model_path_and_labels("en_us")

  assert model_dir != other
  assert check_source_marker(model_dir, _PB_DL_URL)
  assert check_source_marker(other, _OTHER_RELEASE_URL)
  assert (other / "saved_model.pb").read_bytes() == b"another release"
