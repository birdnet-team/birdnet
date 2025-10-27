from __future__ import annotations

from abc import ABC
from typing import Self

import numpy as np

from birdnet.backends import (
  BackendLoader,
  VersionedGeoBackendProtocol,
)
from birdnet.base import SessionBase
from birdnet.geo_models.inference.configs import (
  PredictionConfig,
  RunConfig,
  ScoresConfig,
)
from birdnet.geo_models.inference.prediction_result import PredictionResult
from birdnet.helper import uint_dtype_for


class GeoSessionBase(SessionBase, ABC):
  def __init__(
    self,
    conf: PredictionConfig,
    specific_config: ScoresConfig,
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

    self._backend = self._backend_loader.load_backend(self._conf.processing_conf.device)

    self._is_initialized = True
    return self

  def _run(self, run_config: RunConfig) -> PredictionResult:
    assert self._is_initialized
    assert self._backend is not None

    sample = np.expand_dims(
      np.array(
        [run_config.latitude, run_config.longitude, run_config.week],
        dtype=np.float32,
      ),
      0,
    )

    res = self._backend.predict(sample)
    assert res.dtype == np.float32
    res = res.astype(self._conf.processing_conf.prob_dtype, copy=False)

    res = np.squeeze(res, axis=0)

    n_species = self._conf.model_conf.n_species
    species_ids = np.arange(
      n_species,
      dtype=uint_dtype_for(
        max(0, n_species - 1),
      ),
    )

    invalid_mask = res < self._specific_config.min_confidence
    prediction = PredictionResult(
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
