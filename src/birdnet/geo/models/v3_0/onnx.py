from __future__ import annotations

from birdnet.core.backends import (
  OnnxBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import _GeoSingleFileDownloaderV3_0
from birdnet.globals import (
  MODEL_BACKEND_ONNX,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import ModelInfo

# The geo model input is [latitude, longitude, week] (see GeoSessionBase._run).
_GEO_INPUT_FEATURES = 3
_YEAR_ROUND_WEEK_INPUTS = tuple(float(week) for week in range(1, 49))

models: dict[MODEL_PRECISIONS, ModelInfo] = {
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.onnx",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.onnx",
    dl_size=15503473,
    file_size=15503473,
    sha256="0de81d222c23dcb6fa428e958b4dac978783191357e01b7268a103fc6f08e61a",
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx",
    dl_size=7885053,
    file_size=7885053,
    sha256="80e410f811bf4455e6cb1a2d48e394737d3828fe943a444646778495b67e7bab",
  ),
}


class GeoOnnxDownloaderV3_0(_GeoSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_ONNX
  _models = models


class GeoOnnxBackendFP32V3_0(OnnxBackend, VersionedGeoBackendProtocol):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 0

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def probe_input_size_samples(cls) -> int:
    return _GEO_INPUT_FEATURES

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS


class GeoOnnxBackendFP16V3_0(OnnxBackend, VersionedGeoBackendProtocol):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 0

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def probe_input_size_samples(cls) -> int:
    return _GEO_INPUT_FEATURES

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS
