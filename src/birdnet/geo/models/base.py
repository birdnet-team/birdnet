from abc import abstractmethod
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.core.base import ModelBase
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.globals import GEO_MODEL_VERSIONS


class GeoModelBase(ModelBase):
  def __init__(
    self, model_path: Path, species_list: OrderedSet[str], use_custom_model: bool
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)

  @classmethod
  @abstractmethod
  def get_version(cls) -> GEO_MODEL_VERSIONS: ...

  @abstractmethod
  def predict(self, *args, **kwargs) -> GeoPredictionResult:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_session(self, *args, **kwargs) -> GeoPredictionSession:  # noqa: ANN002, ANN003
    ...
