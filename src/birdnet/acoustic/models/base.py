from abc import ABC, abstractmethod
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
  AcousticSessionBase,
)
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
  def predict(self, *args, **kwargs) -> AcousticPredictionResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_session(self, *args, **kwargs) -> AcousticPredictionSession:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode(self, *args, **kwargs) -> AcousticEncodingResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode_session(self, *args, **kwargs) -> AcousticEncodingSession:  # noqa: ANN002, ANN003
    ...
