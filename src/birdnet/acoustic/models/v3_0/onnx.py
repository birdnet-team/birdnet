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
      "https://zenodo.org/records/24362980/files/BirdNET+_V3.0_Global_12K_FP32.onnx"
    ),
    dl_file_name="BirdNET+_V3.0_Global_12K_FP32.onnx",
    dl_size=345363389,
    file_size=345363389,
    sha256="1486452fd754cbddca267f1ef6f15bd244178f117fd1461a4fbc0b14a8dca223",
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url=(
      "https://zenodo.org/records/24362980/files/BirdNET+_V3.0_Global_12K_FP16.onnx"
    ),
    dl_file_name="BirdNET+_V3.0_Global_12K_FP16.onnx",
    dl_size=173871766,
    file_size=173871766,
    sha256="ae9c42ae50e39f4d70fb1e5a90549456f8357a183450b60afb88d3f1061f8928",
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
