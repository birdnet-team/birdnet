from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import final

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.backends import (
  InferenceBackend,
  PBInferenceBackend,
  check_pb_model_can_be_loaded,
)
from birdnet.geo_models.inference.prediction_result import PredictionResult
from birdnet.geo_models.v2_4.base import GeoDownloaderBaseV2_4, GeoModelBaseV2_4
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_BACKENDS,
  MODEL_PRECISION_FP32,
)
from birdnet.helper import check_protobuf_model_files_exist
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file


class GeoPBDownloaderV2_4(GeoDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path(
      GeoPBModelV2_4.get_model_type(),
      GeoPBModelV2_4.get_version(),
      GeoPBModelV2_4.get_backend(),
      MODEL_PRECISION_FP32,
    )
    lang_dir = get_lang_dir(
      GeoPBModelV2_4.get_model_type(),
      GeoPBModelV2_4.get_version(),
      GeoPBModelV2_4.get_backend(),
    )
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
  def _download_geo_model(cls) -> None:
    dl_url = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
    dl_size = 124522908

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading model",
      )

      print("Extracting models...")
      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_dir = extract_dir / "meta-model"
      species_dl_dir = extract_dir / "labels"

      geo_model_dir, geo_lang_dir = cls._get_paths()
      geo_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(geo_model_dl_dir, geo_model_dir)

      geo_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(species_dl_dir, geo_lang_dir)
      print("Models extracted.")

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_geo_model_available():
      cls._download_geo_model()
    assert cls._check_geo_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class GeoPBModelV2_4(GeoModelBaseV2_4):
  def __init__(
    self, model_path: Path, species_list: OrderedSet[str], use_custom_model: bool
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)

  @classmethod
  @final
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_PB

  @final
  @classmethod
  def get_backend_type(cls) -> type[InferenceBackend]:
    return PBInferenceBackend

  @classmethod
  def load(cls, lang: str) -> GeoPBModelV2_4:
    model_path, species_list = GeoPBDownloaderV2_4.get_model_path_and_labels(lang)
    result = GeoPBModelV2_4(
      model_path=model_path,
      species_list=species_list,
      use_custom_model=False,
    )
    return result

  @classmethod
  def load_custom(
    cls, model: Path, species_list: Path, check_validity: bool
  ) -> GeoPBModelV2_4:
    assert model.is_dir()
    assert species_list.is_file()

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    if not check_protobuf_model_files_exist(model):
      raise ValueError(
        f"Model directory '{model.absolute()}' does not contain the required files for a Protobuf model!"
      )

    if check_validity:
      n_species_in_model = check_pb_model_can_be_loaded(
        model, "serving_default", "MNET_CLASS_ACTIVATION"
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = GeoPBModelV2_4(
      model_path=model, species_list=loaded_species_list, use_custom_model=True
    )

    return result

  def predict(
    self,
    latitude: float,
    longitude: float,
    /,
    *,
    week: int | None = None,
    min_confidence: float = 0.03,
    half_precision: bool = True,
    device: str = "CPU",
  ) -> PredictionResult:
    return super()._predict(
      latitude,
      longitude,
      {
        "model_path": self.model_path,
        "signature_name": "serving_default",
        "prediction_key": "MNET_CLASS_ACTIVATION",
        "input_key": "MNET_INPUT",
      },
      week=week,
      min_confidence=min_confidence,
      device=device,
      half_precision=half_precision,
    )
