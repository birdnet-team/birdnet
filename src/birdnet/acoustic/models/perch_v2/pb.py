from __future__ import annotations

from pathlib import Path
from typing import Literal

from ordered_set import OrderedSet

from birdnet.core.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import check_protobuf_model_files_exist, get_species_from_file

_MIN_TF_VERSION_PERCH_V2 = (2, 20)


def _get_tensorflow_version() -> str:
  try:
    import tensorflow as tf
  except ModuleNotFoundError as e:
    raise RuntimeError(
      "The Perch v2 model requires TensorFlow >= 2.20, but TensorFlow is "
      "not installed."
    ) from e
  return tf.__version__


def _get_major_minor_version(version: str) -> tuple[int, int]:
  version_parts = version.split(".")
  if len(version_parts) < 2:
    raise RuntimeError(f"Could not parse TensorFlow version {version!r}.")

  try:
    return int(version_parts[0]), int(version_parts[1])
  except ValueError as e:
    raise RuntimeError(f"Could not parse TensorFlow version {version!r}.") from e


def check_tf_version_for_perch_v2() -> None:
  version = _get_tensorflow_version()
  if _get_major_minor_version(version) < _MIN_TF_VERSION_PERCH_V2:
    raise RuntimeError(
      "The Perch v2 model requires TensorFlow >= 2.20, "
      f"but {version!r} is installed."
    )


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
