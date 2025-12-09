from __future__ import annotations

from abc import abstractmethod

import numpy as np


class AcousticTensorBase:
  def __init__(self) -> None:
    self._unprocessable_inputs: np.ndarray | None = None

  @property
  @abstractmethod
  def memory_usage_mb(self) -> float: ...

  @abstractmethod
  def write_block(self, *args, **kwargs) -> None: ...

  def set_unprocessable_inputs(self, unprocessable_inputs: np.ndarray) -> None:
    self._unprocessable_inputs = unprocessable_inputs

  @property
  def unprocessable_inputs(self) -> np.ndarray:
    assert self._unprocessable_inputs is not None
    return self._unprocessable_inputs
