from __future__ import annotations

import os
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  PinnedTFBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v2_4.model import (
  _YEAR_ROUND_WEEK_INPUTS,
  GeoDownloaderBaseV2_4,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import ModelInfo, download_zip_members, ensure_download
from birdnet.utils.local_data import get_lang_dir, get_model_path, get_model_root_dir
from birdnet.utils.species_lists import get_species_from_file


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
  def _lock_dir(cls) -> Path:
    return get_model_root_dir("geo", "2.4", "tf") / ".download.lock"

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
    model_path, lang_dir = cls._get_paths()
    download_zip_members(
      cls._model_info.dl_url,
      cls._model_info.dl_size,
      "geo model v2.4 (tf)",
      [(cls._model_info.dl_file_name, model_path), ("labels", lang_dir)],
    )

  @classmethod
  def get_model_path_and_labels(cls, lang: str) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES
    ensure_download(
      cls._check_geo_model_available,
      cls._download_model,
      cls._lock_dir(),
      "the geo model v2.4 (tf) download",
    )
    assert cls._check_geo_model_available()

    model_path, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class GeoTFBackendFP32V2_4(PinnedTFBackend, VersionedGeoBackendProtocol):
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

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS
