from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.backends import (
  TFBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo_models.v2_4.model import GeoDownloaderBaseV2_4
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.helper import (
  ModelInfo,
)
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file


class GeoTFDownloaderV2_4(GeoDownloaderBaseV2_4):
  # All meta models are same for all precisions and int8 is the smallest download
  _model_info = ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_int8.zip",
    dl_file_name="meta-model.tflite",
    dl_size=45948867,
    file_size=29526096,
  )

  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path("geo", "2.4", "tf", MODEL_PRECISION_FP32)
    lang_dir = get_lang_dir("geo", "2.4", "tf")
    return model_path, lang_dir

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    if not model_path.is_file():
      return False

    file_stats = os.stat(model_path)
    is_newest_version = file_stats.st_size == cls._model_info.file_size
    if not is_newest_version:
      return False

    if not lang_dir.is_dir():
      return False

    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls) -> None:
    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        cls._model_info.dl_url,
        zip_download_path,
        download_size=cls._model_info.dl_size,
        description="Downloading geo model v2.4 (tf)",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_path = extract_dir / cls._model_info.dl_file_name
      species_dl_dir = extract_dir / "labels"

      geo_model_path, geo_lang_dir = cls._get_paths()
      geo_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(geo_model_dl_path, geo_model_path)

      geo_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(geo_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, geo_lang_dir)

  @classmethod
  def get_model_path_and_labels(cls, lang: str) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES
    if not cls._check_geo_model_available():
      cls._download_model()
    assert cls._check_geo_model_available()

    model_path, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class GeoTFBackendFP32V2_4(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 62

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
