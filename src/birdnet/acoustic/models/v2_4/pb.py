from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.v2_4.model import (
  AcousticDownloaderBaseV2_4,
)
from birdnet.core.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import (
  download_zip_members,
  ensure_download,
  saved_model_is_current,
  write_source_marker,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path, get_model_root_dir
from birdnet.utils.species_lists import get_species_from_file

_PB_DL_URL = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
_PB_DL_SIZE = 124522908


class AcousticPBDownloaderV2_4(AcousticDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path("acoustic", "2.4", "pb", MODEL_PRECISION_FP32)
    lang_dir = get_lang_dir("acoustic", "2.4", "pb")
    return model_path, lang_dir

  @classmethod
  def _lock_dir(cls) -> Path:
    return get_model_root_dir("acoustic", "2.4", "pb") / ".download.lock"

  @classmethod
  def _check_acoustic_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()
    if not saved_model_is_current(model_path, _PB_DL_URL):
      return False
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls) -> None:
    model_dir, lang_dir = cls._get_paths()
    download_zip_members(
      _PB_DL_URL,
      _PB_DL_SIZE,
      "acoustic model v2.4 (pb)",
      [("audio-model", model_dir), ("labels", lang_dir)],
    )
    write_source_marker(model_dir, _PB_DL_URL)

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    ensure_download(
      cls._check_acoustic_model_available,
      cls._download_model,
      cls._lock_dir(),
      "the acoustic model v2.4 (pb) download",
    )
    assert cls._check_acoustic_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class AcousticPBBackendFP32V2_4(PBBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def input_key(cls) -> str:
    return "inputs"

  @classmethod
  def prediction_signature_name(cls) -> str:
    return "basic"

  @classmethod
  def prediction_key(cls) -> str:
    return "scores"

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_signature_name(cls) -> str | None:
    return "embeddings"

  @classmethod
  def encoding_key(cls) -> str | None:
    return "embeddings"

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32


class AcousticRavenBackendFP32V2_4(PBBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def name(cls) -> str:
    return f"{MODEL_BACKEND_PB}-raven"

  @classmethod
  def applies_sigmoid(cls) -> bool:
    # The Raven export bakes the sigmoid into the graph on both bases.
    return True

  @classmethod
  def input_key(cls) -> str:
    return "inputs"

  @classmethod
  def prediction_signature_name(cls) -> str:
    return "basic"

  @classmethod
  def prediction_key(cls) -> str:
    return "scores"

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_signature_name(cls) -> str | None:
    return None

  @classmethod
  def encoding_key(cls) -> str | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
