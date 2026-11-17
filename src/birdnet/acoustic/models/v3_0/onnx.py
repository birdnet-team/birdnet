from __future__ import annotations

from birdnet.acoustic.models.v3_0.model import (
  AcousticModelV3_0,
  _AcousticSingleFileDownloaderV3_0,
)
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
      "https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP32.onnx"
    ),
    dl_file_name="BirdNET+_V3.0_Global_12K_FP32.onnx",
    dl_size=345249691,
    file_size=345249691,
    sha256="e78c3b50e224cd79d549d3c3b5efde2436acd1d3fbd72694f2a0b8547fa18685",
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url=(
      "https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP16.onnx"
    ),
    dl_file_name="BirdNET+_V3.0_Global_12K_FP16.onnx",
    dl_size=173377757,
    file_size=173377757,
    sha256="9891d615f12a7a51fbc15dc40ffbd65a841dae9a2d9a80874ddc892729796dd3",
  ),
}


class AcousticOnnxDownloaderV3_0(_AcousticSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_ONNX
  _models = models


class _AcousticOnnxBackendV3_0(OnnxBackend, VersionedAcousticBackendProtocol):
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
    return AcousticModelV3_0.get_segment_size_samples()

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True


class AcousticOnnxBackendFP32V3_0(_AcousticOnnxBackendV3_0):
  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32


class AcousticOnnxBackendFP16V3_0(_AcousticOnnxBackendV3_0):
  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16
