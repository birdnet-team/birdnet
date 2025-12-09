from pathlib import Path
from typing import Any

from ordered_set import OrderedSet

from birdnet.backends import VersionedGeoBackendProtocol
from birdnet.geo_models.inference.configs import (
  InferenceConfig,
  ModelConfig,
  PredictionConfig,
  ProcessingConfig,
  RunConfig,
)
from birdnet.geo_models.inference.prediction_result import GeoPredictionResult
from birdnet.geo_models.inference.session import GeoSessionBase
from birdnet.globals import GEO_MODEL_VERSIONS


class GeoPredictionSession(GeoSessionBase):
  def __init__(
    self,
    species_list: OrderedSet[str],
    model_path: Path,
    model_is_custom: bool,
    model_version: GEO_MODEL_VERSIONS,
    model_backend_type: type[VersionedGeoBackendProtocol],
    model_backend_custom_kwargs: dict[str, Any],
    *,
    min_confidence: float,
    half_precision: bool,
    device: str,
  ) -> None:
    assert len(species_list) > 0
    assert model_path.exists()
    assert model_backend_custom_kwargs is not None

    half_precision = ProcessingConfig.validate_half_precision(half_precision)

    min_confidence = PredictionConfig.validate_min_confidence(min_confidence)

    device = ProcessingConfig.validate_device(device)

    super().__init__(
      conf=InferenceConfig(
        model_conf=ModelConfig(
          species_list=species_list,
          path=model_path,
          is_custom=model_is_custom,
          version=model_version,
          backend_type=model_backend_type,
          backend_kwargs=model_backend_custom_kwargs,
        ),
        processing_conf=ProcessingConfig(
          half_precision=half_precision,
          device=ProcessingConfig.validate_device(device),
        ),
      ),
      specific_config=PredictionConfig(
        min_confidence=min_confidence,
      ),
    )

  def run(
    self,
    latitude: float,
    longitude: float,
    /,
    *,
    week: int | None = None,
  ) -> GeoPredictionResult:
    latitude = RunConfig.validate_latitude(latitude)
    longitude = RunConfig.validate_longitude(longitude)
    week = RunConfig.validate_week(week)

    return self._run(
      RunConfig(
        latitude=latitude,
        longitude=longitude,
        week=week,
      ),
    )
