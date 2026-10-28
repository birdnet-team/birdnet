from __future__ import annotations

from pathlib import Path

from birdnet.acoustic.models.v3_0.model import (
  AcousticModelV3_0,
  _AcousticSingleFileDownloaderV3_0,
)
from birdnet.core.backends import TorchBackend, VersionedAcousticBackendProtocol
from birdnet.globals import MODEL_BACKEND_PT, MODEL_PRECISION_FP32, MODEL_PRECISIONS
from birdnet.utils.downloads import ModelInfo

models: dict[MODEL_PRECISIONS, ModelInfo] = {
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


class AcousticPTDownloaderV3_0(_AcousticSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_PT
  _models = models

  # The single precision stays callable without an argument.
  @classmethod
  def _get_paths(
    cls, precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32
  ) -> tuple[Path, Path]:
    return super()._get_paths(precision)

  @classmethod
  def _check_acoustic_model_available(
    cls, precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32
  ) -> bool:
    return super()._check_acoustic_model_available(precision)

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32) -> None:
    return super()._download_model(precision)


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
    return AcousticModelV3_0.get_segment_size_samples()

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32

  @classmethod
  def applies_sigmoid(cls) -> bool:
    # The TorchScript export applies the activation itself (prediction_needs_sigmoid
    # is False), so its output is a score like that of every other V3.0 export.
    return True
