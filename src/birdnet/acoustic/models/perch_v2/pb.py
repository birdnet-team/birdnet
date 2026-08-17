from __future__ import annotations

import shutil
import tempfile
import zipfile
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
from birdnet.utils.helper import (
  check_protobuf_model_files_exist,
  check_source_marker,
  download_file_tqdm,
  get_species_from_file,
  write_source_marker,
)
from birdnet.utils.local_data import APP_DIR

_MIN_TF_VERSION_PERCH_V2 = (2, 20)


def _get_tensorflow_version() -> str:
  try:
    import tensorflow as tf
  except ModuleNotFoundError as e:
    raise RuntimeError(
      "The Perch v2 model requires TensorFlow >= 2.20, but TensorFlow is not installed."
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
      f"The Perch v2 model requires TensorFlow >= 2.20, but {version!r} is installed."
    )


class AcousticPBDownloaderPerchV2:
  MODEL_DOWNLOAD_URL_CPU = "https://tuc.cloud/index.php/s/z3eo89G9MmHexG6/download"
  MODEL_DOWNLOAD_SIZE_CPU = 379116813
  MODEL_DOWNLOAD_URL_GPU = "https://tuc.cloud/index.php/s/HddMnr9Lf4wdAYJ/download"
  MODEL_DOWNLOAD_SIZE_GPU = 379119624
  LABELS_HEADER = "inat2024_fsd50k"

  @classmethod
  def _get_model_root(cls) -> Path:
    return APP_DIR / "acoustic-models" / "perch-v2"

  @classmethod
  def _get_paths(
    cls,
    device: Literal["CPU", "GPU"],
  ) -> tuple[Path, Path]:
    assert device in ("CPU", "GPU")
    device_name = device.lower()
    model_path = cls._get_model_root() / f"perch-v2-{device_name}"
    labels_path = model_path / "assets" / "labels.csv"
    return model_path, labels_path

  @classmethod
  def _get_download_info(cls, device: Literal["CPU", "GPU"]) -> tuple[str, int]:
    assert device in ("CPU", "GPU")
    if device == "CPU":
      return cls.MODEL_DOWNLOAD_URL_CPU, cls.MODEL_DOWNLOAD_SIZE_CPU
    return cls.MODEL_DOWNLOAD_URL_GPU, cls.MODEL_DOWNLOAD_SIZE_GPU

  @classmethod
  def _download_model(cls, device: Literal["CPU", "GPU"]) -> None:
    dl_url, dl_size = cls._get_download_info(device)

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description=f"Downloading Perch v2 model ({device.lower()})",
      )

      print("Extracting...")  # noqa: T201
      extract_dir = Path(temp_dir) / "extracted"
      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      model_path, _ = cls._get_paths(device)
      model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(model_path, ignore_errors=True)
      shutil.move(extract_dir, model_path)
      write_source_marker(model_path, dl_url)
      print("Extracted.")  # noqa: T201

  @classmethod
  def _check_acoustic_model_available(cls, device: Literal["CPU", "GPU"]) -> bool:
    model_path, labels_path = cls._get_paths(device)
    dl_url, _ = cls._get_download_info(device)

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)
    model_is_downloaded &= check_source_marker(model_path, dl_url)
    model_is_downloaded &= labels_path.is_file()

    return model_is_downloaded

  @classmethod
  def get_model_path_and_labels(
    cls, device: Literal["CPU", "GPU"]
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_acoustic_model_available(device):
      cls._download_model(device)
    assert cls._check_acoustic_model_available(device)

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
