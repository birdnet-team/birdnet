from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

from ordered_set import OrderedSet

from birdnet.core.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
  _get_major_minor_version,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import (
  download_zip_members,
  ensure_download,
  saved_model_is_current,
  write_source_marker,
)
from birdnet.utils.local_data import get_birdnet_app_data_folder
from birdnet.utils.logging_utils import suppress_native_stderr
from birdnet.utils.species_lists import get_species_from_file

_MIN_TF_VERSION_PERCH_V2 = (2, 20)


def _get_tensorflow_version() -> str:
  try:
    with suppress_native_stderr():
      import tensorflow as tf
  except ModuleNotFoundError as e:
    raise RuntimeError(
      "The Perch v2 model requires TensorFlow >= 2.20, but TensorFlow is not installed."
    ) from e
  return tf.__version__


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
    return get_birdnet_app_data_folder() / "acoustic-models" / "perch-v2"

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
  def _lock_dir(cls) -> Path:
    return cls._get_model_root() / ".download.lock"

  @classmethod
  def _download_model(cls, device: Literal["CPU", "GPU"]) -> None:
    dl_url, dl_size = cls._get_download_info(device)
    model_path, _ = cls._get_paths(device)
    download_zip_members(
      dl_url, dl_size, f"Perch v2 model ({device.lower()})", [(None, model_path)]
    )
    write_source_marker(model_path, dl_url)

  @classmethod
  def _check_acoustic_model_available(cls, device: Literal["CPU", "GPU"]) -> bool:
    model_path, labels_path = cls._get_paths(device)
    dl_url, _ = cls._get_download_info(device)
    return saved_model_is_current(model_path, dl_url) and labels_path.is_file()

  @classmethod
  def get_model_path_and_labels(
    cls, device: Literal["CPU", "GPU"]
  ) -> tuple[Path, OrderedSet[str]]:
    ensure_download(
      partial(cls._check_acoustic_model_available, device),
      partial(cls._download_model, device),
      cls._lock_dir(),
      f"the Perch v2 model ({device.lower()}) download",
    )
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
