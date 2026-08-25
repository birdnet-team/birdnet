from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.core.backends import TorchBackend, VersionedAcousticBackendProtocol
from birdnet.globals import MODEL_BACKEND_PT, MODEL_PRECISION_FP32, MODEL_PRECISIONS
from birdnet.utils.helper import (
  ModelInfo,
  ensure_single_file_model,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path

models = {
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url=(
      "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.pt"
    ),
    dl_file_name="BirdNET+_V3.0-preview3.1_Global_11K_FP32.pt",
    dl_size=541831823,
    file_size=541831823,
    sha256="c7d3b1506a579a50c306a8cfd620b6a122d074d4d52169d88de242fa8390b825",
  )
}


class AcousticPTDownloaderV3_0(AcousticDownloaderBaseV3_0):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path(
      "acoustic",
      "3.0",
      MODEL_BACKEND_PT,
      MODEL_PRECISION_FP32,
      content_tag=models[MODEL_PRECISION_FP32].content_tag,
    )
    lang_dir = get_lang_dir("acoustic", "3.0", MODEL_BACKEND_PT)
    return model_path, lang_dir

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return cls._get_paths()[1]

  @classmethod
  def _check_acoustic_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()
    if not model_path.is_file():
      return False
    if model_path.stat().st_size != models[MODEL_PRECISION_FP32].file_size:
      return False
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls) -> None:
    model_path, _lang_dir = cls._get_paths()
    ensure_single_file_model(
      models[MODEL_PRECISION_FP32],
      model_path,
      legacy_path=get_model_path(
        "acoustic", "3.0", MODEL_BACKEND_PT, MODEL_PRECISION_FP32
      ),
      description="Downloading acoustic model v3.0 (pt, fp32)",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for acoustic pt model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()
    if not cls._check_acoustic_model_available():
      cls._download_model()
    assert cls._check_acoustic_model_available()

    model_path, _lang_dir = cls._get_paths()
    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class AcousticPTBackendFP32V3_0(TorchBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 0

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 1

  @classmethod
  def probe_input_size_samples(cls) -> int:
    return 96_000

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
