from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from birdnet_v2.base import ModelBase


class AcousticInferenceBackend(ABC):
  @abstractmethod
  def lazy_load(self, device_name: str) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray) -> np.ndarray: ...


class AcousticModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()

  @abstractmethod
  def get_backend_instance(self) -> AcousticInferenceBackend: ...

  @abstractmethod
  def get_backend_type(self) -> type[AcousticInferenceBackend]: ...

  @abstractmethod
  def get_backend_args(self) -> dict: ...
