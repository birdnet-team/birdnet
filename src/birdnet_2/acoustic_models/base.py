from abc import abstractmethod
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_PRECISIONS
from birdnet2.base import ModelBase


class AcousticModelBase(ModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)

  @classmethod
  @abstractmethod
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS: ...
