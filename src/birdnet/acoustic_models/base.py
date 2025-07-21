import multiprocessing
from abc import ABC, abstractmethod

import numpy as np

from birdnet.base import ACOUSTIC_MODEL_VERSIONS, MODEL_PRECISIONS, ModelBase


class AcousticInferenceBackend(ABC):
  @abstractmethod
  def load(self) -> None: ...

  @abstractmethod
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray: ...

  @classmethod
  @abstractmethod
  def supports_cow(cls) -> bool: ...


class AcousticInferenceBackendLoader:
  def __init__(
    self,
    backend_type: type[AcousticInferenceBackend],
    backend_kwargs: dict,
  ) -> None:
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs
    self._backend: AcousticInferenceBackend | None = None

  def _load_backend(self) -> AcousticInferenceBackend:
    assert self._backend is None
    backend = self._backend_type(**self._backend_kwargs)
    backend.load()
    self._backend = backend
    return backend

  def on_before_worker_initialized(self) -> None:
    if (
      multiprocessing.get_start_method() == "fork" and self._backend_type.supports_cow()
    ):
      self._load_backend()

  def load_backend(self) -> AcousticInferenceBackend:
    if self._backend is None:
      return self._load_backend()
    assert self._backend is not None
    return self._backend

  @property
  def backend(self) -> AcousticInferenceBackend:
    assert self._backend is not None
    return self._backend


class AcousticModelBase(ModelBase):
  def __init__(self) -> None:
    super().__init__()
    self._precision: MODEL_PRECISIONS | None = None

  @classmethod
  @abstractmethod
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS: ...

  @property
  def precision(self) -> MODEL_PRECISIONS:
    """
    Returns the precision of the model.
    """
    assert self._precision is not None
    return self._precision

  @classmethod
  @abstractmethod
  def get_inference_backend_type(cls) -> type[AcousticInferenceBackend]: ...

  @abstractmethod
  def get_inference_backend_args(self) -> dict: ...
