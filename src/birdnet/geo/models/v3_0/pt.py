from __future__ import annotations

from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.backends import (
  TorchBackend,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
from birdnet.globals import (
  MODEL_BACKEND_PT,
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

# The release also contains BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.pt, which is a
# training checkpoint (a state dict), not a loadable module. The TorchScript export
# is the one to use here.
models = {
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_TorchScript.pt",
    dl_file_name="BirdNET+_Geomodel_V3.0.4_Global_14K_TorchScript.pt",
    dl_size=15261167,
    file_size=15261167,
    sha256="80e93ff8886a481107c47651e35abbf6c5799992cb1943999062bad0bf1a9c93",
  ),
}


class GeoPTDownloaderV3_0(GeoDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("geo", "3.0", MODEL_BACKEND_PT)

  @classmethod
  def _get_model_path(cls) -> Path:
    return get_model_path(
      "geo",
      "3.0",
      MODEL_BACKEND_PT,
      MODEL_PRECISION_FP32,
      content_tag=models[MODEL_PRECISION_FP32].content_tag,
    )

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path = cls._get_model_path()

    if not model_path.is_file():
      return False

    if model_path.stat().st_size != models[MODEL_PRECISION_FP32].file_size:
      return False

    return cls._check_labels_available()

  @classmethod
  def _download_model(cls) -> None:
    ensure_single_file_model(
      models[MODEL_PRECISION_FP32],
      cls._get_model_path(),
      legacy_path=get_model_path("geo", "3.0", MODEL_BACKEND_PT, MODEL_PRECISION_FP32),
      description="Downloading geo model v3.0 (pt, fp32)",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    if precision not in models:
      raise ValueError(
        f"Unsupported model precision for geo pt model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()

    if not cls._check_geo_model_available():
      cls._download_model()
    assert cls._check_geo_model_available()

    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return cls._get_model_path(), labels


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
    # returns the same probabilities.
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
