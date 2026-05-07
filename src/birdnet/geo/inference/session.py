from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Self

import numpy as np
from ordered_set import OrderedSet

from birdnet.core.backends import (
  BackendLoader,
  VersionedGeoBackendProtocol,
)
from birdnet.core.base import SessionBase
from birdnet.geo.inference.configs import (
  InferenceConfig,
  ModelConfig,
  PredictionConfig,
  ProcessingConfig,
  RunConfig,
)
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.globals import (
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  GEO_YEAR_ROUND_AGGREGATION_MAX,
  GEO_YEAR_ROUND_AGGREGATIONS,
)
from birdnet.utils.helper import get_uint_dtype


class GeoSessionBase(SessionBase, ABC):
  def __init__(
    self,
    conf: InferenceConfig,
    specific_config: PredictionConfig,
  ) -> None:
    self._conf = conf
    self._specific_config = specific_config
    self._backend: VersionedGeoBackendProtocol | None = None
    self._is_initialized = False
    super().__init__()

  def __enter__(self) -> Self:
    assert not self._is_initialized

    self._backend_loader = BackendLoader(
      model_path=self._conf.model_conf.path,
      backend_type=self._conf.model_conf.backend_type,
      backend_kwargs=self._conf.model_conf.backend_kwargs,
    )

    self._backend = self._backend_loader.load_backend(
      self._conf.processing_conf.device, self._conf.processing_conf.half_precision
    )

    self._is_initialized = True
    return self

  def _run(self, run_config: RunConfig) -> GeoPredictionResult:
    assert self._is_initialized
    assert self._backend is not None

    if run_config.week is None:
      if self._conf.model_conf.version == GEO_MODEL_VERSION_V2_4:
        # v2.4 was trained with -1 as the year-round sentinel
        sample = np.expand_dims(
          np.array(
            [run_config.latitude, run_config.longitude, -1.0],
            dtype=np.float32,
          ),
          0,
        )
        res = self._backend.predict(sample)
        res = np.squeeze(res, axis=0)
      else:
        # v3.0+: feed all 48 weeks as a batch and aggregate
        samples = np.array(
          [
            [run_config.latitude, run_config.longitude, float(w)]
            for w in range(1, 49)
          ],
          dtype=np.float32,
        )  # shape: (48, 3)
        res_batch = self._backend.predict(samples)  # shape: (48, n_species)
        if run_config.year_round_aggregation == GEO_YEAR_ROUND_AGGREGATION_MAX:
          res = np.max(res_batch, axis=0)
        else:
          res = np.mean(res_batch, axis=0)
      result_week = -1
    else:
      sample = np.expand_dims(
        np.array(
          [run_config.latitude, run_config.longitude, run_config.week],
          dtype=np.float32,
        ),
        0,
      )
      res = self._backend.predict(sample)
      res = np.squeeze(res, axis=0)
      result_week = run_config.week

    n_species = self._conf.model_conf.n_species
    species_ids = np.arange(
      n_species,
      dtype=get_uint_dtype(
        max(0, n_species - 1),
      ),
    )

    invalid_mask = res < self._specific_config.min_confidence
    prediction = GeoPredictionResult(
      model_path=self._conf.model_conf.path,
      model_version=self._conf.model_conf.version,
      model_precision=self._backend.precision(),
      latitude=run_config.latitude,
      longitude=run_config.longitude,
      week=result_week,
      species_list=self._conf.model_conf.species_list,
      species_probs=res,
      species_ids=species_ids,
      species_masked=invalid_mask,
    )

    return prediction

  def __exit__(self, *args) -> None:
    assert self._is_initialized
    self._backend = None
    self._backend_loader = None
    self._is_initialized = False


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
    year_round_aggregation: GEO_YEAR_ROUND_AGGREGATIONS = GEO_YEAR_ROUND_AGGREGATION_MAX,  # noqa: E501
  ) -> GeoPredictionResult:
    latitude = RunConfig.validate_latitude(latitude)
    longitude = RunConfig.validate_longitude(longitude)
    week = RunConfig.validate_week(week)
    year_round_aggregation = RunConfig.validate_year_round_aggregation(
      year_round_aggregation
    )

    return self._run(
      RunConfig(
        latitude=latitude,
        longitude=longitude,
        week=week,
        year_round_aggregation=year_round_aggregation,
      ),
    )
