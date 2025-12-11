from abc import ABC, abstractmethod
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.result_base import SessionBase
from birdnet.acoustic_models.session import AcousticSessionBase
from birdnet.core.base import ModelBase, ResultBase
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS


class AcousticModelBase(ModelBase, ABC):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)

  @classmethod
  @abstractmethod
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS: ...

  @abstractmethod
  def encode(self, *args, **kwargs) -> ResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode_session(self, *args, **kwargs) -> AcousticSessionBase:  # noqa: ANN002, ANN003
    ...
