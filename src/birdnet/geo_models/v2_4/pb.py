from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.backends import (
  PBBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo_models.v2_4.model import GeoDownloaderBaseV2_4
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.helper import check_protobuf_model_files_exist
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file


class GeoPBDownloaderV2_4(GeoDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path("geo", "2.4", "pb", MODEL_PRECISION_FP32)
    lang_dir = get_lang_dir("geo", "2.4", "pb")
    return model_path, lang_dir

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)

    model_is_downloaded &= lang_dir.is_dir()
    for lang in cls.AVAILABLE_LANGUAGES:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()

    return model_is_downloaded

  @classmethod
  def _download_model(cls) -> None:
    dl_url = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
    dl_size = 124522908

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading geo model v2.4 (pb)",
      )

      print("Extracting...")
      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_dir = extract_dir / "meta-model"
      species_dl_dir = extract_dir / "labels"

      geo_model_dir, geo_lang_dir = cls._get_paths()
      geo_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(geo_model_dir, ignore_errors=True)
      shutil.move(geo_model_dl_dir, geo_model_dir)

      geo_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(geo_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, geo_lang_dir)
      print("Extracted.")

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_geo_model_available():
      cls._download_model()
    assert cls._check_geo_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class GeoPBBackendFP32V2_4(PBBackend, VersionedGeoBackendProtocol):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
  ) -> None:
    super().__init__(model_path, device_name, half_precision)

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
