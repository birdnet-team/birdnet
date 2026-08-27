from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  TFBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
from birdnet.globals import (
  LIBRARY_LITERT,
  LIBRARY_TYPES,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import (
  ModelInfo,
  ensure_single_file_model,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path
from birdnet.utils.logging_utils import suppress_native_stderr

_SUPPORTED_TF_VERSIONS = ("2.18", "2.19")
_YEAR_ROUND_WEEK_INPUTS = tuple(float(week) for week in range(1, 49))


def check_tf_library_for_v3_0(library: LIBRARY_TYPES) -> None:
  if library == LIBRARY_LITERT:
    raise RuntimeError(
      "The geo model v3.0 TF backend is not supported with ai_edge_litert. "
      "Use library='tflite' or another backend, e.g. load('geo', '3.0', 'pb', ...) "
      "or, without TensorFlow, load('geo', '3.0', 'onnx', ...) / 'pt'."
    )


def check_tf_runtime_compatibility_for_v3_0(library: LIBRARY_TYPES) -> None:
  check_tf_library_for_v3_0(library)
  _check_tf_version_for_v3_0()


def _check_tf_version_for_v3_0() -> None:
  with suppress_native_stderr():
    import tensorflow as tf

  version: str = tf.__version__
  if not any(version.startswith(v) for v in _SUPPORTED_TF_VERSIONS):
    supported = " or ".join(_SUPPORTED_TF_VERSIONS)
    raise RuntimeError(
      f"The geo model v3.0 TF backend requires TensorFlow {supported}, "
      f"but {version!r} is installed. "
      "Consider using the PB backend instead: load('geo', '3.0', 'pb', ...)."
    )


models = {
  MODEL_PRECISION_INT8: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_INT8.tflite",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_INT8.tflite",
    dl_size=4244200,
    file_size=4244200,
    sha256="58997b3656c8d8d8efb51308bcab54fd78d16b2ecbf93d9199051c0c622bca08",
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.tflite",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.tflite",
    dl_size=7705320,
    file_size=7705320,
    sha256="bee9d29461404c76f812798747288f40d6e0143ce72dd825872a5b8785732e1f",
  ),
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.tflite",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.tflite",
    dl_size=15317232,
    file_size=15317232,
    sha256="bffdfb18454a250205bd85240a1054eb5f43be5c419653d3979ef292c2e4ab3f",
  ),
}


class GeoTFDownloaderV3_0(GeoDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("geo", "3.0", "tf")

  @classmethod
  def _get_model_path(cls, precision: MODEL_PRECISIONS) -> Path:
    return get_model_path(
      "geo", "3.0", "tf", precision, content_tag=models[precision].content_tag
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
      legacy_path=get_model_path("geo", "3.0", "tf", precision),
      description=f"Downloading geo model v3.0 (tf, {precision})",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
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


class GeoTFBackendFP32V3_0(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  def load(self) -> None:
    check_tf_runtime_compatibility_for_v3_0(self._inference_library)
    super().load()

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 515

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS


class GeoTFBackendFP16V3_0(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  def load(self) -> None:
    check_tf_runtime_compatibility_for_v3_0(self._inference_library)
    super().load()

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 586

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS


class GeoTFBackendInt8V3_0(TFBackend, VersionedGeoBackendProtocol):
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  def load(self) -> None:
    check_tf_runtime_compatibility_for_v3_0(self._inference_library)
    super().load()

  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 515

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_INT8

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return _YEAR_ROUND_WEEK_INPUTS
