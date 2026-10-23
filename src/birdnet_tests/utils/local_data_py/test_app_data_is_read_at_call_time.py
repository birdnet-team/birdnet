"""The app data directory is looked up when a path is needed, not at import.

``BIRDNET_APP_DATA`` set after ``import birdnet`` has to move every cached
artifact at once; a path frozen at import would leave the models in one
directory and their labels in another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import birdnet.acoustic.models.v3_0.model as acoustic_model
import birdnet.geo.models.v3_0.model as geo_model
import birdnet.utils.taxonomy_v3 as taxonomy_v3
from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
from birdnet.geo.models.v2_4.pb import GeoPBDownloaderV2_4
from birdnet.geo.models.v2_4.tf import GeoTFDownloaderV2_4
from birdnet.geo.models.v3_0.pb import GeoPBDownloaderV3_0
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.local_data import (
  get_benchmark_dir,
  get_lang_dir,
  get_model_path,
)

pytestmark = pytest.mark.no_tf


def test_every_cache_path_follows_the_directory_set_after_import(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  app_data = tmp_path / "app-data"
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(app_data))

  paths = {
    "model": get_model_path("acoustic", "3.0", "onnx", "fp32"),
    "lang dir": get_lang_dir("geo", "3.0", "tf"),
    "benchmarks": get_benchmark_dir("acoustic", "probe"),
    "perch v2": AcousticPBDownloaderPerchV2._get_model_root(),
    "acoustic v3.0 labels": acoustic_model._labels_raw_path(),
    "acoustic v3.0 legacy labels": acoustic_model._legacy_labels_raw_path(),
    "acoustic v3.0 lock": acoustic_model._setup_lock_dir(),
    "geo v3.0 labels": geo_model._labels_raw_path(),
    "geo v3.0 legacy labels": geo_model._legacy_labels_raw_path(),
    "geo v3.0 lock": geo_model._setup_lock_dir(),
    "taxonomy": taxonomy_v3.get_taxonomy_v3_input().path,
    "legacy taxonomy": taxonomy_v3._legacy_taxonomy_v3_path(),
    "taxonomy lock": taxonomy_v3._taxonomy_v3_lock_dir(),
    "acoustic 2.4 pb download lock": AcousticPBDownloaderV2_4._lock_dir(),
    "acoustic 2.4 tf download lock": AcousticTFDownloaderV2_4._lock_dir(),
    "acoustic 3.0 pb download lock": AcousticPBDownloaderV3_0._lock_dir(),
    "geo 2.4 pb download lock": GeoPBDownloaderV2_4._lock_dir(),
    "geo 2.4 tf download lock": GeoTFDownloaderV2_4._lock_dir(),
    "geo 3.0 pb download lock": GeoPBDownloaderV3_0._lock_dir(),
    "perch v2 download lock": AcousticPBDownloaderPerchV2._lock_dir(),
  }

  stray = {
    name: path
    for name, path in paths.items()
    if not path.is_relative_to(app_data.resolve())
  }
  assert not stray, f"not under {app_data}: {stray}"
