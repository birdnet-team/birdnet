from abc import abstractmethod
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.base import ModelBase
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_PRECISIONS


class AcousticModelBase(ModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)
    self._precision = precision

  @classmethod
  @abstractmethod
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS: ...

  @property
  def precision(self) -> MODEL_PRECISIONS:
    """
    Returns the precision of the model.
    """
    return self._precision  # type: ignore
