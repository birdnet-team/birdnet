from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  PBBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v2_4.model import (
  _YEAR_ROUND_WEEK_INPUTS,
  GeoDownloaderBaseV2_4,
  GeoModelV2_4,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import (
  download_zip_members,
  ensure_download,
  saved_model_is_current,
  write_source_marker,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path, get_model_root_dir
from birdnet.utils.species_lists import read_official_labels

_PB_DL_URL = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
_PB_DL_SIZE = 124522908


class GeoPBDownloaderV2_4(GeoDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path("geo", "2.4", "pb", MODEL_PRECISION_FP32)
    lang_dir = get_lang_dir("geo", "2.4", "pb")
    return model_path, lang_dir

  @classmethod
  def _lock_dir(cls) -> Path:
    return get_model_root_dir("geo", "2.4", "pb") / ".download.lock"

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()
    if not saved_model_is_current(model_path, _PB_DL_URL):
      return False
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls) -> None:
    model_dir, lang_dir = cls._get_paths()
    download_zip_members(
      _PB_DL_URL,
      _PB_DL_SIZE,
      "geo model v2.4 (pb)",
      [("meta-model", model_dir), ("labels", lang_dir)],
    )
    write_source_marker(model_dir, _PB_DL_URL)

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    ensure_download(
      cls._check_geo_model_available,
      cls._download_model,
      cls._lock_dir(),
      "the geo model v2.4 (pb) download",
    )
    assert cls._check_geo_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = read_official_labels(
      lang_file, GeoModelV2_4.get_n_classes(), "geo model v2.4"
    )
    return model_dir, labels


class GeoPBBackendFP32V2_4(PBBackend, VersionedGeoBackendProtocol):
  @classmethod
  def input_key(cls) -> str:
    return "MNET_INPUT"

  @classmethod
  def prediction_signature_name(cls) -> str:
    return "serving_default"

  @classmethod
  def prediction_key(cls) -> str:
    return "MNET_CLASS_ACTIVATION"

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_signature_name(cls) -> str | None:
    return None

  @classmethod
  def encoding_key(cls) -> str | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS
