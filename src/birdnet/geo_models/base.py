from abc import ABC, abstractmethod

import numpy as np

from birdnet.base import ModelBase


class GeoInferenceBackend(ABC):
  @abstractmethod
  def load(self) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray: ...


class GeoModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()
