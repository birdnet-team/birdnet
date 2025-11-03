import time
from abc import ABC, abstractmethod
from multiprocessing import current_process
from pathlib import Path
from threading import current_thread
from typing import Self

from ordered_set import OrderedSet

from birdnet.helper import get_hash


class PredictionResultBase(ABC):
  @abstractmethod
  def save(self, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003

  @classmethod
  @abstractmethod
  def load(cls, *args, **kwargs) -> Self: ...  # noqa: ANN002, ANN003

  @property
  @abstractmethod
  def memory_size_mb(self) -> float: ...


class SessionBase(ABC):
  def __init__(self) -> None:
    self._session_id = get_session_id()

  @abstractmethod
  def __enter__(self) -> Self: ...

  @abstractmethod
  def __exit__(self, *args): ...

  @abstractmethod
  def run(self, *args, **kwargs) -> PredictionResultBase: ...


def get_session_id() -> str:
  """
  Get a unique session ID based on the current process and thread.

  Example for two processes (fork):
    Process 1: 53554_127397175535424_1762165676846175803
    Process 2: 53555_127397175535424_1762165676846559511

  Example for two processes (spawn):
    Process 1: 54155_126834937165632_1762165717644505438
    Process 2: 54154_132842492557120_1762165717644777865

  Example for two threads in the same process:
    Thread 1: 53142_138235445503680_1762165643891762916
    Thread 2: 53142_138235453896384_1762165653498085145

  Example for same thread and process but different calls:
    Call 1: 50179_128078941120320_1762165462208616340
    Call 2: 50179_128078941120320_1762165485281125126
  """
  proc = current_process()
  thread = current_thread()
  timestamp = time.time_ns()
  result = f"{proc.ident}_{thread.ident}_{timestamp}"
  return result


def get_session_id_hash(session_id: str) -> str:
  hash_digest = get_hash(session_id)[:5]
  return hash_digest


class ModelBase(ABC):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
  ) -> None:
    super().__init__()
    self._model_path = model_path
    self._species_list = species_list
    self._use_custom_model = use_custom_model

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

  @classmethod
  @abstractmethod
  def predict_session(cls, *args, **kwargs) -> SessionBase:  # noqa: ANN002, ANN003
    ...
