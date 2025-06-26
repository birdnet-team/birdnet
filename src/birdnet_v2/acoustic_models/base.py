from abc import ABC, abstractmethod

import numpy as np

from birdnet_v2.base import ModelBase


class AcousticInferenceBackend(ABC):
  @abstractmethod
  def lazy_load(self) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray) -> np.ndarray: ...


class AcousticModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()

  @abstractmethod
  def get_backend_instance(self) -> AcousticInferenceBackend: ...
