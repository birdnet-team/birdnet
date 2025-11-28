from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

from ordered_set import OrderedSet

from birdnet.acoustic_models.v2_4.model import (
  AcousticDownloaderBaseV2_4,
)
from birdnet.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.helper import check_protobuf_model_files_exist
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file


class AcousticPBDownloaderPerchV2:
  MODEL_HANDLE_CUDA = "google/bird-vocalization-classifier/tensorFlow2/perch_v2"
  MODEL_HANDLE_CPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
  LABELS_HEADER = "inat2024_fsd50k"

  @classmethod
  def _get_paths(cls, cuda: bool) -> tuple[Path, Path]:
    import kagglehub

    model_handle = cls.MODEL_HANDLE_CUDA if cuda else cls.MODEL_HANDLE_CPU
    model_path = Path(kagglehub.model_download(model_handle))
    labels_path = model_path / "assets" / "labels.csv"
    return model_path, labels_path

  @classmethod
  def _check_acoustic_model_available(cls, cuda: bool) -> bool:
    model_path, model_path = cls._get_paths(cuda)

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)
    model_is_downloaded &= model_path.is_file()

    return model_is_downloaded

  @classmethod
  def get_model_path_and_labels(cls, cuda: bool) -> tuple[Path, OrderedSet[str]]:
    cls._check_acoustic_model_available(cuda)

    model_dir, labels_path = cls._get_paths(cuda)
    labels = get_species_from_file(labels_path, encoding="utf8")
    labels.remove(cls.LABELS_HEADER)
    assert len(labels) == 14795
    return model_dir, labels


class AcousticPBBackendFP32PerchV2(PBBackend, VersionedAcousticBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def input_key(cls) -> str:
    return "inputs"

  @classmethod
  def scores_signature_name(cls) -> str:
    return "serving_default"

  @classmethod
  def scores_prediction_key(cls) -> str:
    return "label"

  @classmethod
  def emb_supported(cls) -> bool:
    return True

  @classmethod
  def emb_signature_name(cls) -> str | None:
    return "serving_default"

  @classmethod
  def emb_prediction_key(cls) -> str | None:
    return "embedding"

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
