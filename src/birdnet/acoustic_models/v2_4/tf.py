from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic_models.v2_4.model import (
  AcousticDownloaderBaseV2_4,
)
from birdnet.backends import (
  TFBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_BACKEND_TF,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet.helper import ModelInfo
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file

models = {
  MODEL_PRECISION_INT8: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_int8.zip",
    dl_file_name="audio-model-int8.tflite",
    dl_size=45948867,
    file_size=41064296,
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_fp16.zip",
    dl_file_name="audio-model-fp16.tflite",
    dl_size=53025528,
    file_size=25932528,
  ),
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip",
    dl_file_name="audio-model.tflite",
    dl_size=76822925,
    file_size=51726412,
  ),
}


class AcousticTFDownloaderV2_4(AcousticDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_path = get_model_path(
      "acoustic",
      "2.4",
      MODEL_BACKEND_TF,
      precision,
    )
    lang_dir = get_lang_dir(
      "acoustic",
      "2.4",
      MODEL_BACKEND_TF,
    )
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
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
        description=f"Downloading acoustic model v2.4 (tf, {precision})",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dl_path = extract_dir / models[precision].dl_file_name
      species_dl_dir = extract_dir / "labels"

      acoustic_model_path, acoustic_lang_dir = cls._get_paths(precision)
      acoustic_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_path, acoustic_model_path)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(acoustic_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES
    if not cls._check_acoustic_model_available(precision):
      cls._download_model(precision)
    assert cls._check_acoustic_model_available(precision)

    model_path, langs_path = cls._get_paths(precision)

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class AcousticTFBackendInt8V2_4(TFBackend, VersionedAcousticBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 643

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 640

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_INT8


class AcousticTFBackendFP16V2_4(TFBackend, VersionedAcousticBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 546

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 545

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16


class AcousticTFBackendFP32V2_4(TFBackend, VersionedAcousticBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 546

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 545

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
