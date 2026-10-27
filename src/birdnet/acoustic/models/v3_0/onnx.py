from __future__ import annotations

from birdnet.acoustic.models.v3_0.model import _AcousticSingleFileDownloaderV3_0
from birdnet.core.backends import OnnxBackend, VersionedAcousticBackendProtocol
from birdnet.globals import (
  MODEL_BACKEND_ONNX,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import ModelInfo

models: dict[MODEL_PRECISIONS, ModelInfo] = {
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


class AcousticOnnxDownloaderV3_0(_AcousticSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_ONNX
  _models = models


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

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True


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

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True
