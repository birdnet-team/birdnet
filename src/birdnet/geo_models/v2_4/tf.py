from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.backends import (
  InferenceBackend,
  TFInferenceBackend,
  check_tf_model_can_be_loaded,
)
from birdnet.geo_models.inference.prediction_result import PredictionResult
from birdnet.geo_models.v2_4.base import GeoDownloaderBaseV2_4, GeoModelBaseV2_4
from birdnet.globals import (
  LIBRARY_TYPES,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FP32,
)
from birdnet.helper import (
  ModelInfo,
)
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file

if TYPE_CHECKING:
  pass

MODEL_LOGITS_IDX = 62

# All meta models are same for all precisions and int8 is the smallest download
model_info = ModelInfo(
  dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_int8.zip",
  dl_file_name="meta-model.tflite",
  dl_size=45948867,
  file_size=29526096,
)


class GeoTFDownloaderV2_4(GeoDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path(
      GeoTFModelV2_4.get_model_type(),
      GeoTFModelV2_4.get_version(),
      GeoTFModelV2_4.get_backend(),
      MODEL_PRECISION_FP32,
    )
    lang_dir = get_lang_dir(
      GeoTFModelV2_4.get_model_type(),
      GeoTFModelV2_4.get_version(),
      GeoTFModelV2_4.get_backend(),
    )
    return model_path, lang_dir

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    if not model_path.is_file():
      return False

    file_stats = os.stat(model_path)
    is_newest_version = file_stats.st_size == model_info.file_size
    if not is_newest_version:
      return False

    if not lang_dir.is_dir():
      return False

    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_geo_model(cls) -> None:
    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        model_info.dl_url,
        zip_download_path,
        download_size=model_info.dl_size,
        description="Downloading model",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_path = extract_dir / model_info.dl_file_name
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
      cls._download_geo_model()
    assert cls._check_geo_model_available()

    model_path, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class GeoTFModelV2_4(GeoModelBaseV2_4):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
    library: LIBRARY_TYPES,
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)
    self._library = library

  @final
  @classmethod
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_TF

  @final
  @classmethod
  def get_backend_type(cls) -> type[InferenceBackend]:
    return TFInferenceBackend

  @classmethod
  def load(
    cls,
    lang: MODEL_LANGUAGES,
    library: LIBRARY_TYPES,
  ) -> GeoTFModelV2_4:
    model_path, species_list = GeoTFDownloaderV2_4.get_model_path_and_labels(lang)
    result = GeoTFModelV2_4(
      model_path, species_list, use_custom_model=False, library=library
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model: Path,
    species_list: Path,
    check_validity: bool,
    library: LIBRARY_TYPES,
  ) -> GeoTFModelV2_4:
    assert model.is_file()
    assert species_list.is_file()

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    if check_validity:
      n_species_in_model = check_tf_model_can_be_loaded(
        model, library, out_idx=MODEL_LOGITS_IDX
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = GeoTFModelV2_4(
      model, loaded_species_list, use_custom_model=True, library=library
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
  ) -> PredictionResult:
    return super()._predict(
      latitude,
      longitude,
      {
        "model_path": self.model_path,
        "inference_library": self._library,
        "in_idx": 0,
        "out_idx": MODEL_LOGITS_IDX,
      },
      week=week,
      min_confidence=min_confidence,
      device="CPU",
      half_precision=half_precision,
    )
