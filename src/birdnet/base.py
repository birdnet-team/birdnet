from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.backends import InferenceBackend
from birdnet.globals import MODEL_BACKENDS, MODEL_PRECISIONS


class PredictionResultBase:
  @abstractmethod
  def save(self, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003

  @classmethod
  @abstractmethod
  def load(cls, *args, **kwargs) -> Self: ...  # noqa: ANN002, ANN003

  @property
  @abstractmethod
  def memory_size_mb(self) -> float: ...


class ModelBase(ABC):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
  ) -> None:
    super().__init__()
    self._model_path = model_path
    self._species_list = species_list
    self._use_custom_model = use_custom_model
    self._precision: MODEL_PRECISIONS = precision

  @classmethod
  @abstractmethod
  def get_backend(cls) -> MODEL_BACKENDS: ...

  @classmethod
  @abstractmethod
  def get_backend_type(cls) -> type[InferenceBackend]: ...

  @property
  def model_path(self) -> Path:
    return self._model_path

  @property
  def species_list(self) -> OrderedSet[str]:
    return self._species_list

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @property
  def use_custom_model(self) -> bool:
    return self._use_custom_model

  @property
  def precision(self) -> MODEL_PRECISIONS:
    return self._precision

  @classmethod
  @abstractmethod
  def load(cls, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
    ...

  @classmethod
  @abstractmethod
  def load_custom(cls, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
    ...

  @classmethod
  @abstractmethod
  def predict(cls, *args, **kwargs) -> PredictionResultBase:  # noqa: ANN002, ANN003
    ...
