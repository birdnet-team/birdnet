from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.core.backends import OnnxBackend, VersionedAcousticBackendProtocol
from birdnet.globals import (
  MODEL_BACKEND_ONNX,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import (
  ModelInfo,
  ensure_single_file_model,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path

models = {
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url=(
      "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.onnx"
    ),
    dl_file_name="BirdNET+_V3.0-preview3.1_Global_11K_FP32.onnx",
    dl_size=541598502,
    file_size=541598502,
    sha256="999bb627e8954516d6a222eb76438b896fafc3791c2d10e72f02147ccd562c05",
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url=(
      "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP16.onnx"
    ),
    dl_file_name="BirdNET+_V3.0-preview3.1_Global_11K_FP16.onnx",
    dl_size=271554018,
    file_size=271554018,
    sha256="b42679c722f87f48dd19ccfb846e56db3dfa7e2bb3d846fba121d978b5a27170",
  ),
}


class AcousticOnnxDownloaderV3_0(AcousticDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("acoustic", "3.0", MODEL_BACKEND_ONNX)

  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_path = get_model_path(
      "acoustic",
      "3.0",
      MODEL_BACKEND_ONNX,
      precision,
      content_tag=models[precision].content_tag,
    )
    lang_dir = get_lang_dir("acoustic", "3.0", MODEL_BACKEND_ONNX)
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path, lang_dir = cls._get_paths(precision)
    if not model_path.is_file():
      return False
    if model_path.stat().st_size != models[precision].file_size:
      return False
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS) -> None:
    model_path, _lang_dir = cls._get_paths(precision)
    ensure_single_file_model(
      models[precision],
      model_path,
      legacy_path=get_model_path("acoustic", "3.0", MODEL_BACKEND_ONNX, precision),
      description=f"Downloading acoustic model v3.0 (onnx, {precision})",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()
    if not cls._check_acoustic_model_available(precision):
      cls._download_model(precision)
    assert cls._check_acoustic_model_available(precision)

    model_path, _ = cls._get_paths(precision)
    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class AcousticOnnxBackendFP32V3_0(OnnxBackend, VersionedAcousticBackendProtocol):
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


class AcousticOnnxBackendFP16V3_0(OnnxBackend, VersionedAcousticBackendProtocol):
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
    return MODEL_PRECISION_FP16
