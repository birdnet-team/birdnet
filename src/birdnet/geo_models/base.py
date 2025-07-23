from abc import abstractmethod
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.base import ModelBase
from birdnet.globals import GEO_MODEL_VERSIONS


class GeoModelBase(ModelBase):
  def __init__(
    self, model_path: Path, species_list: OrderedSet[str], use_custom_model: bool
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)

  @classmethod
  @abstractmethod
  def get_version(cls) -> GEO_MODEL_VERSIONS: ...
