from __future__ import annotations

from abc import abstractmethod


class TensorBase:
  @property
  @abstractmethod
  def memory_usage_mb(self) -> float: ...

  @abstractmethod
  def write_block(self, *args, **kwargs) -> None: ...
