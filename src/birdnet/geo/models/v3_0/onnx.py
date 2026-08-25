from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  OnnxBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
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

# The geo model input is [latitude, longitude, week] (see GeoSessionBase._run).
_GEO_INPUT_FEATURES = 3
_YEAR_ROUND_WEEK_INPUTS = tuple(float(week) for week in range(1, 49))

models = {
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


class GeoOnnxDownloaderV3_0(GeoDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("geo", "3.0", MODEL_BACKEND_ONNX)

  @classmethod
  def _get_model_path(cls, precision: MODEL_PRECISIONS) -> Path:
    return get_model_path(
      "geo",
      "3.0",
      MODEL_BACKEND_ONNX,
      precision,
      content_tag=models[precision].content_tag,
    )

  @classmethod
  def _check_geo_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path = cls._get_model_path(precision)

    if not model_path.is_file():
      return False

    if model_path.stat().st_size != models[precision].file_size:
      return False

    return cls._check_labels_available()

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS) -> None:
    ensure_single_file_model(
      models[precision],
      cls._get_model_path(precision),
      legacy_path=get_model_path("geo", "3.0", MODEL_BACKEND_ONNX, precision),
      description=f"Downloading geo model v3.0 (onnx, {precision})",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    if precision not in models:
      raise ValueError(
        f"Unsupported model precision for geo onnx model: {precision}. "
        f"Currently supported precisions are: {', '.join(models)}."
      )
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()

    if not cls._check_geo_model_available(precision):
      cls._download_model(precision)
    assert cls._check_geo_model_available(precision)

    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return cls._get_model_path(precision), labels


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
