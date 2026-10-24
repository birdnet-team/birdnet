from __future__ import annotations

from pathlib import Path

from birdnet.core.backends import (
  TorchBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import _GeoSingleFileDownloaderV3_0
from birdnet.globals import (
  MODEL_BACKEND_PT,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import ModelInfo

# The geo model input is [latitude, longitude, week] (see GeoSessionBase._run).
_GEO_INPUT_FEATURES = 3
_YEAR_ROUND_WEEK_INPUTS = tuple(float(week) for week in range(1, 49))

# The release also contains BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.pt, which is a
# training checkpoint (a state dict), not a loadable module. The TorchScript export
# is the one to use here.
models: dict[MODEL_PRECISIONS, ModelInfo] = {
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_TorchScript.pt",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_TorchScript.pt",
    dl_size=15261167,
    file_size=15261167,
    sha256="80e93ff8886a481107c47651e35abbf6c5799992cb1943999062bad0bf1a9c93",
  ),
}


class GeoPTDownloaderV3_0(_GeoSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_PT
  _models = models

  # The single precision stays callable without an argument.
  @classmethod
  def _get_model_path(cls, precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32) -> Path:
    return super()._get_model_path(precision)

  @classmethod
  def _check_geo_model_available(
    cls, precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32
  ) -> bool:
    return super()._check_geo_model_available(precision)

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32) -> None:
    return super()._download_model(precision)


class GeoPTBackendFP32V3_0(TorchBackend, VersionedGeoBackendProtocol):
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
  def prediction_needs_sigmoid(cls) -> bool:
    # The TorchScript export ends at the linear layer, while the TFLite, ProtoBuf
    # and ONNX exports all end in a sigmoid. Apply it here so that every backend
    # returns the same scores.
    #
    # This backend also serves load_custom("geo", "3.0", "pt", ...), so a custom
    # TorchScript geo model is expected to return logits as well - one that
    # applies the activation itself would be squashed a second time.
    return True

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS
