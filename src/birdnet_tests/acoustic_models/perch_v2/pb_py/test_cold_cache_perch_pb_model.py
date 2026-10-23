"""A cold cache: one download, the model in place, the marker written."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.helper import check_source_marker
from birdnet_tests.helper import fake_zip_downloads, saved_model_members

pytestmark = pytest.mark.no_tf

_N_LABELS = 14795


def test_cold_cache_downloads_once_and_marks_the_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  url = AcousticPBDownloaderPerchV2.MODEL_DOWNLOAD_URL_CPU
  species = [f"Species_{index}" for index in range(_N_LABELS)]
  labels_csv = "\n".join([AcousticPBDownloaderPerchV2.LABELS_HEADER, *species, ""])
  archive = saved_model_members("") | {"assets/labels.csv": labels_csv.encode()}
  calls = fake_zip_downloads(
    monkeypatch, {url: archive}, held_lock=AcousticPBDownloaderPerchV2._lock_dir()
  )

  model_dir, labels = AcousticPBDownloaderPerchV2.get_model_path_and_labels("CPU")
  AcousticPBDownloaderPerchV2.get_model_path_and_labels("CPU")

  assert calls == [url]
  assert model_dir == AcousticPBDownloaderPerchV2._get_paths("CPU")[0]
  assert check_source_marker(model_dir, url)
  assert list(labels) == species
  assert not AcousticPBDownloaderPerchV2._lock_dir().exists()
