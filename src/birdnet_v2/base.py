import multiprocessing as mp
from abc import ABC, abstractmethod

import birdnet_v2.logging_utils as bn_logging


class ModelBase(ABC):
  def __init__(self, version: str, backend: str, model_type: str) -> None:
    self._model_type = model_type
    self._backend = backend
    self._version = version

  @property
  def version(self) -> str:
    return self._version

  @property
  def model_type(self) -> str:
    return self._model_type

  @property
  def backend(self) -> str:
    return self._backend

