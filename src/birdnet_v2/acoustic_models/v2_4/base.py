from abc import ABC, abstractmethod

from birdnet_v2.acoustic_models.base import AcousticModelBase


class AcousticModelBaseV2_4(AcousticModelBase):
  def __init__(self, backend: str) -> None:
    super().__init__("v2.4", backend)
    self._sig_fmin: int = 0
    self._sig_fmax: int = 15_000
    self._sample_rate: int = 48_000
    self._chunk_size_s: float = 3.0
    self._chunk_size_samples = int(self._chunk_size_s * self._sample_rate)

  @property
  def sig_fmin(self) -> int:
    return self._sig_fmin

  @property
  def sig_fmax(self) -> int:
    return self._sig_fmax

  @property
  def sample_rate(self) -> int:
    return self._sample_rate

  @property
  def chunk_size_s(self) -> float:
    return self._chunk_size_s

  @property
  def chunk_size_samples(self) -> int:
    return self._chunk_size_samples
