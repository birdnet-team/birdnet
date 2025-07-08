from abc import ABC, abstractmethod

import numpy as np

from birdnet.base import ModelBase
from birdnet.io_lock import IOLockHandler


class AcousticInferenceBackend(ABC):
  @abstractmethod
  def lazy_load(self, device_name: str, io_lock_handler: IOLockHandler) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray) -> np.ndarray: ...


class AcousticModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()

  @classmethod
  @abstractmethod
  def get_backend_type(cls) -> type[AcousticInferenceBackend]: ...

  @abstractmethod
  def get_backend_args(self) -> dict: ...
