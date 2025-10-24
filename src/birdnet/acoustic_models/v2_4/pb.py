from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.backends import (
  PBInferenceBackend,
  VersionedAcousticInferenceBackendProtocol,
  check_pb_model_can_be_loaded,
)
from birdnet.acoustic_models.v2_4.model import (
  AcousticDownloaderBaseV2_4,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
)
from birdnet.helper import check_protobuf_model_files_exist
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file


class AcousticPBDownloaderV2_4(AcousticDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path("acoustic", "2.4", "pb", MODEL_PRECISION_FP32)
    lang_dir = get_lang_dir("acoustic", "2.4", "pb")
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)

    model_is_downloaded &= lang_dir.is_dir()
    for lang in cls.AVAILABLE_LANGUAGES:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()

    return model_is_downloaded

  @classmethod
  def _download_acoustic_model(cls) -> None:
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

      acoustic_model_dl_dir = extract_dir / "audio-model"
      species_dl_dir = extract_dir / "labels"

      acoustic_model_dir, acoustic_lang_dir = cls._get_paths()
      acoustic_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_dir, acoustic_model_dir)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)
      print("Models extracted.")

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_acoustic_model_available():
      cls._download_acoustic_model()
    assert cls._check_acoustic_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class PBAcousticInferenceBackendV2_4(
  PBInferenceBackend, VersionedAcousticInferenceBackendProtocol
):
  def __init__(
    self,
    model_path: Path,
    inference_strategy: Literal["scores", "embeddings"],
    device_name: str,
  ) -> None:
    if inference_strategy == "scores":
      signature_name = "basic"
      prediction_key = "scores"
      input_key = "inputs"
    elif inference_strategy == "embeddings":
      signature_name = "serving_default"
      prediction_key = "EMBEDDING_OUTPUT"
      input_key = "MNET_INPUT"
    else:
      raise AssertionError()

    super().__init__(
      model_path,
      signature_name,
      prediction_key,
      input_key,
      device_name,
    )

  @classmethod
  def check_model_can_be_loaded(
    cls,
    model_path: Path,
    **kwargs: Any,
  ) -> int | None:
    n_outputs = check_pb_model_can_be_loaded(
      model_path,
      "basic",
      "scores",
    )
    return n_outputs
