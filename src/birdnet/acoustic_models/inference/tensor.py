from __future__ import annotations

from abc import abstractmethod

import numpy as np

from birdnet.helper import get_uint_dtype


class AcousticTensorBase:
  def __init__(self) -> None:
    self._unprocessable_inputs: np.ndarray | None = None

  @property
  @abstractmethod
  def memory_usage_mb(self) -> float: ...

  @abstractmethod
  def write_block(self, *args, **kwargs) -> None: ...

  def set_unprocessable_inputs(self, unprocessable_inputs: set[int]) -> None:
    if len(unprocessable_inputs) == 0:
      unprocessable_inputs_np = np.array([], dtype=np.uint8)
    else:
      unprocessable_inputs_np = np.array(
        sorted(unprocessable_inputs), dtype=get_uint_dtype(max(unprocessable_inputs))
      )
    self._unprocessable_inputs = unprocessable_inputs_np

  @property
  def unprocessable_inputs(self) -> np.ndarray:
    assert self._unprocessable_inputs is not None
    return self._unprocessable_inputs
