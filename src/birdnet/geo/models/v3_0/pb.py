from __future__ import annotations

import shutil
import tempfile
import zipfile
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
from birdnet.utils.helper import (
  check_protobuf_model_files_exist,
  download_file_tqdm,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path

_YEAR_ROUND_WEEK_INPUTS = tuple(float(week) for week in range(1, 49))


class GeoPBDownloaderV3_0(GeoDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("geo", "3.0", "pb")

  @classmethod
  def _get_model_path(cls) -> Path:
    return get_model_path("geo", "3.0", "pb", MODEL_PRECISION_FP32)

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path = cls._get_model_path()

    if not model_path.is_dir():
      return False
    if not check_protobuf_model_files_exist(model_path):
      return False

    return cls._check_labels_available()

  @classmethod
  def _download_model(cls) -> None:
    dl_url = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.2/BirdNET+_Geomodel_V3.0.2_Global_12K_FP32_TF.zip"
    dl_size = 13466042

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading geo model v3.0 (pb)",
      )

      print("Extracting...")
      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_dir = extract_dir / "BirdNET+_Geomodel_V3.0.2_Global_12K_FP32_TF"
      geo_model_dir = cls._get_model_path()
      geo_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(geo_model_dir, ignore_errors=True)
      shutil.move(geo_model_dl_dir, geo_model_dir)
      print("Extracted.")

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()

    if not cls._check_geo_model_available():
      cls._download_model()
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
