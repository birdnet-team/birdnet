from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  PBBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import (
  download_zip_members,
  ensure_download,
  saved_model_is_current,
)
from birdnet.utils.helper import get_species_from_file, write_source_marker
from birdnet.utils.local_data import get_lang_dir, get_model_path, get_model_root_dir

_YEAR_ROUND_WEEK_INPUTS = tuple(float(week) for week in range(1, 49))

_PB_DL_URL = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_FP32_TF.zip"
_PB_DL_SIZE = 14196836


class GeoPBDownloaderV3_0(GeoDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("geo", "3.0", "pb")

  @classmethod
  def _get_model_path(cls) -> Path:
    return get_model_path("geo", "3.0", "pb", MODEL_PRECISION_FP32)

  @classmethod
  def _lock_dir(cls) -> Path:
    return get_model_root_dir("geo", "3.0", "pb") / ".download.lock"

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    if not saved_model_is_current(cls._get_model_path(), _PB_DL_URL):
      return False
    return cls._check_labels_available()

  @classmethod
  def _download_model(cls) -> None:
    model_dir = cls._get_model_path()
    download_zip_members(
      _PB_DL_URL,
      _PB_DL_SIZE,
      "geo model v3.0 (pb)",
      [("BirdNET+_Geomodel_V3.0.4_Global_14K_FP32_TF", model_dir)],
    )
    write_source_marker(model_dir, _PB_DL_URL)

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()

    ensure_download(
      cls._check_geo_model_available,
      cls._download_model,
      cls._lock_dir(),
      "the geo model v3.0 (pb) download",
    )
    assert cls._check_geo_model_available()

    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return cls._get_model_path(), labels


class GeoPBBackendFP32V3_0(PBBackend, VersionedGeoBackendProtocol):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
  ) -> None:
    super().__init__(model_path, device_name, half_precision)

  @classmethod
  def input_key(cls) -> str:
    return "input"

  @classmethod
  def prediction_signature_name(cls) -> str:
    return "serving_default"

  @classmethod
  def prediction_key(cls) -> str:
    return "probabilities"

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
