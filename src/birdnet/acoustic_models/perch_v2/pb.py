from __future__ import annotations

from pathlib import Path
from typing import Literal

from ordered_set import OrderedSet

from birdnet.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.helper import check_protobuf_model_files_exist
from birdnet.utils import get_species_from_file


class AcousticPBDownloaderPerchV2:
  MODEL_HANDLE_CUDA = "google/bird-vocalization-classifier/tensorFlow2/perch_v2"
  MODEL_HANDLE_CPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
  LABELS_HEADER = "inat2024_fsd50k"

  @classmethod
  def _get_paths(cls, device: Literal["CPU", "GPU"]) -> tuple[Path, Path]:
    import kagglehub

    assert device in ("CPU", "GPU")
    model_handle = cls.MODEL_HANDLE_CPU if device == "CPU" else cls.MODEL_HANDLE_CUDA
    model_path = Path(kagglehub.model_download(model_handle))
    labels_path = model_path / "assets" / "labels.csv"
    return model_path, labels_path

  @classmethod
  def _check_acoustic_model_available(cls, device: Literal["CPU", "GPU"]) -> bool:
    model_path, model_path = cls._get_paths(device)

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)
    model_is_downloaded &= model_path.is_file()

    return model_is_downloaded

  @classmethod
  def get_model_path_and_labels(
    cls, device: Literal["CPU", "GPU"]
  ) -> tuple[Path, OrderedSet[str]]:
    cls._check_acoustic_model_available(device)

    model_dir, labels_path = cls._get_paths(device)
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
  def prediction_signature_name(cls) -> str:
    return "serving_default"

  @classmethod
  def prediction_key(cls) -> str:
    return "label"

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_signature_name(cls) -> str | None:
    return "serving_default"

  @classmethod
  def encoding_key(cls) -> str | None:
    return "embedding"

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
