from abc import ABC, abstractmethod

import numpy as np

from birdnet.base import GEO_MODEL_VERSIONS, ModelBase


class GeoInferenceBackend(ABC):
  @abstractmethod
  def load(self) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray: ...


class GeoModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()

  @classmethod
  @abstractmethod
  def get_version(cls) -> GEO_MODEL_VERSIONS: ...
