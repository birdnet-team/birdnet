from abc import ABC, abstractmethod

import numpy as np

from birdnet.base import MODEL_PRECISIONS, ModelBase
from birdnet.io_lock import IOLockHandler


class AcousticInferenceBackend(ABC):
  @abstractmethod
  def load(self, io_lock_handler: IOLockHandler) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray: ...


class AcousticModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()
    self._precision: MODEL_PRECISIONS | None = None

  @property
  def precision(self) -> MODEL_PRECISIONS:
    """
    Returns the precision of the model.
    """
    assert self._precision is not None
    return self._precision

  @classmethod
  @abstractmethod
  def get_backend_type(cls) -> type[AcousticInferenceBackend]: ...

  @abstractmethod
  def get_backend_args(self) -> dict: ...
