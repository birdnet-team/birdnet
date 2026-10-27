from abc import abstractmethod
from pathlib import Path
from typing import Any, Self

from ordered_set import OrderedSet

from birdnet.core.backends import BackendLoader, VersionedGeoBackendProtocol
from birdnet.core.base import ModelBase
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.globals import (
  GEO_MODEL_VERSIONS,
  GEO_YEAR_ROUND_AGGREGATION_MAX,
  GEO_YEAR_ROUND_AGGREGATIONS,
)
from birdnet.utils.species_lists import validate_species_list


class GeoModelBase(ModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    is_custom_model: bool,
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    super().__init__(model_path, species_list, is_custom_model)
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs

  @property
  def backend_type(self) -> type[VersionedGeoBackendProtocol]:
    return self._backend_type

  @property
  def backend_kwargs(self) -> dict[str, Any]:
    return self._backend_kwargs

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> Self:
    return cls(
      model_path,
      species_list,
      is_custom_model=False,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> Self:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list = validate_species_list(species_list)

    if check_validity:
      n_species_in_model = BackendLoader.check_model_can_be_loaded(
        model_path, backend_type, backend_kwargs
      )

      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, "
          f"but species list '{species_list.absolute()}' "
          f"has {len(loaded_species_list)} species!"
        )

    return cls(
      model_path,
      loaded_species_list,
      is_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )

  @classmethod
  @abstractmethod
  def get_version(cls) -> GEO_MODEL_VERSIONS: ...

  def predict_session(
    self,
    /,
    *,
    min_confidence: float = 0.03,
    half_precision: bool = False,
    device: str = "CPU",
  ) -> GeoPredictionSession:
    return GeoPredictionSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_is_custom=self.is_custom_model,
      model_version=self.get_version(),
      model_backend_type=self.backend_type,
      model_backend_custom_kwargs=self.backend_kwargs,
      min_confidence=min_confidence,
      half_precision=half_precision,
      device=device,
    )

  def predict(
    self,
    latitude: float,
    longitude: float,
    /,
    *,
    week: int | None = None,
    year_round_aggregation: GEO_YEAR_ROUND_AGGREGATIONS = GEO_YEAR_ROUND_AGGREGATION_MAX,  # noqa: E501
    min_confidence: float = 0.03,
    half_precision: bool = False,
    device: str = "CPU",
  ) -> GeoPredictionResult:
    with self.predict_session(
      min_confidence=min_confidence,
      half_precision=half_precision,
      device=device,
    ) as session:
      return session.run(
        latitude,
        longitude,
        week=week,
        year_round_aggregation=year_round_aggregation,
      )
