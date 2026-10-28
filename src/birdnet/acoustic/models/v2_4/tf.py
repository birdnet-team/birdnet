from __future__ import annotations

import os
from collections.abc import Collection
from functools import partial
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.custom_tflite import (
  LEGACY_V2_4_PREDICTION_INDICES,
  CustomTFLiteBackend,
)
from birdnet.acoustic.models.v2_4.model import (
  AcousticDownloaderBaseV2_4,
  AcousticModelV2_4,
)
from birdnet.core.backends import (
  PinnedTFBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKEND_TF,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import ModelInfo, download_zip_members, ensure_download
from birdnet.utils.local_data import get_lang_dir, get_model_path, get_model_root_dir
from birdnet.utils.species_lists import read_official_labels

models: dict[MODEL_PRECISIONS, ModelInfo] = {
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
  def _lock_dir(cls) -> Path:
    # One lock for all precisions: they share the labels directory.
    return get_model_root_dir("acoustic", "2.4", MODEL_BACKEND_TF) / ".download.lock"

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
    model_path, lang_dir = cls._get_paths(precision)
    members: list[tuple[str | None, Path]] = [
      (models[precision].dl_file_name, model_path)
    ]
    # The precisions share the labels directory. While a sibling is usable, its
    # labels are being read without the lock and stay in place; replacing them
    # would pull the directory from under such a load.
    sibling_usable = any(
      cls._check_acoustic_model_available(other)
      for other in models
      if other != precision
    )
    if not sibling_usable:
      members.append(("labels", lang_dir))
    download_zip_members(
      models[precision].dl_url,
      models[precision].dl_size,
      f"acoustic model v2.4 (tf, {precision})",
      members,
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES
    ensure_download(
      partial(cls._check_acoustic_model_available, precision),
      partial(cls._download_model, precision),
      cls._lock_dir(),
      f"the acoustic model v2.4 (tf, {precision}) download",
    )
    assert cls._check_acoustic_model_available(precision)

    model_path, langs_path = cls._get_paths(precision)

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = read_official_labels(
      lang_file, AcousticModelV2_4.get_n_classes(), "acoustic model v2.4"
    )
    return model_path, labels


class _AcousticTFBackendV2_4(PinnedTFBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def supports_encoding(cls) -> bool:
    return True


class AcousticTFBackendInt8V2_4(_AcousticTFBackendV2_4):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 643

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 640

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_INT8


class AcousticTFBackendFP16V2_4(_AcousticTFBackendV2_4):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 546

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 545

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16


class AcousticTFBackendFP32V2_4(_AcousticTFBackendV2_4):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 546

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 545

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32


class AcousticTFBackendCustomV2_4(
  CustomTFLiteBackend, VersionedAcousticBackendProtocol
):
  """A custom 2.4 classifier, whichever head and export version produced it."""

  @classmethod
  def model_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return "2.4"

  @classmethod
  def legacy_prediction_indices(cls) -> Collection[int]:
    return tuple(LEGACY_V2_4_PREDICTION_INDICES)


class AcousticTFBackendCustomNoEmbeddingsV2_4(AcousticTFBackendCustomV2_4):
  """A custom 2.4 classifier whose file exposes no embeddings tensor."""

  @classmethod
  def supports_encoding(cls) -> bool:
    return False
