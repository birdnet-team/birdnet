from abc import abstractmethod
from pathlib import Path
from typing import Any

from ordered_set import OrderedSet

from birdnet.core.backends import VersionedGeoBackendProtocol
from birdnet.core.base import ModelBase
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.globals import GEO_MODEL_VERSIONS


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
  @abstractmethod
  def get_version(cls) -> GEO_MODEL_VERSIONS: ...

  @abstractmethod
  def predict(self, *args, **kwargs) -> GeoPredictionResult:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_session(self, *args, **kwargs) -> GeoPredictionSession:  # noqa: ANN002, ANN003
    ...
