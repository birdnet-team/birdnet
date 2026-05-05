from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  TFBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
from birdnet.globals import (
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import (
  ModelInfo,
  download_file_tqdm,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path

# TODO: update with real zenodo URLs and file sizes once published
models = {
  MODEL_PRECISION_INT8: ModelInfo(
    dl_url="TODO",
    dl_file_name="TODO",
    dl_size=0,
    file_size=0,
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url="TODO",
    dl_file_name="TODO",
    dl_size=0,
    file_size=0,
  ),
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="TODO",
    dl_file_name="TODO",
    dl_size=0,
    file_size=0,
  ),
}


class GeoTFDownloaderV3_0(GeoDownloaderBaseV3_0):
  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_path = get_model_path("geo", "3.0", "tf", precision)
    lang_dir = get_lang_dir("geo", "3.0", "tf")
    return model_path, lang_dir

  @classmethod
  def _check_geo_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path, lang_dir = cls._get_paths(precision)

    if not model_path.is_file():
      return False

    file_stats = os.stat(model_path)
    is_newest_version = file_stats.st_size == models[precision].file_size
    if not is_newest_version:
      return False

    if not lang_dir.is_dir():
      return False

    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS) -> None:
    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        models[precision].dl_url,
        zip_download_path,
        download_size=models[precision].dl_size,
        description=f"Downloading geo model v3.0 (tf, {precision})",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_path = extract_dir / models[precision].dl_file_name
      species_dl_dir = extract_dir / "labels"

      geo_model_path, geo_lang_dir = cls._get_paths(precision)
      geo_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(geo_model_dl_path, geo_model_path)

      geo_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(geo_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, geo_lang_dir)

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES
    if not cls._check_geo_model_available(precision):
      cls._download_model(precision)
    assert cls._check_geo_model_available(precision)

    model_path, langs_path = cls._get_paths(precision)

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class GeoTFBackendFP32V3_0(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 0  # TODO: update with real tensor index

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32


class GeoTFBackendFP16V3_0(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 0  # TODO: update with real tensor index

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16


class GeoTFBackendInt8V3_0(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 0  # TODO: update with real tensor index

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_INT8
