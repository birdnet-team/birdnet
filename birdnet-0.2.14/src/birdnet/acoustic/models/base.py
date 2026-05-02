from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet.core.backends import VersionedAcousticBackendProtocol
from birdnet.core.base import ModelBase
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS


class AcousticModelBase(ModelBase, ABC):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    is_custom_model: bool,
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    super().__init__(model_path, species_list, is_custom_model)
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs

  @property
  def backend_type(self) -> type[VersionedAcousticBackendProtocol]:
    return self._backend_type

  @property
  def backend_kwargs(self) -> dict[str, Any]:
    return self._backend_kwargs

  @classmethod
  @abstractmethod
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:  # noqa: ANN002
    """Return the string label that identifies the acoustic model version.

    Returns:
      ACOUSTIC_MODEL_VERSIONS: Registered enum constant for the supported version.
    """
    ...

  @classmethod
  @abstractmethod
  def get_sig_fmin(cls) -> int: ...

  @classmethod
  @abstractmethod
  def get_sig_fmax(cls) -> int: ...

  @classmethod
  @abstractmethod
  def get_sample_rate(cls) -> int: ...

  @classmethod
  @abstractmethod
  def get_segment_size_s(cls) -> float: ...

  @classmethod
  @abstractmethod
  def get_segment_size_samples(cls) -> int: ...

  @abstractmethod
  def predict(self, *args, **kwargs) -> AcousticPredictionResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_arrays(self, *args, **kwargs) -> AcousticPredictionResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_session(self, *args, **kwargs) -> AcousticPredictionSession:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode(self, *args, **kwargs) -> AcousticEncodingResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode_arrays(self, *args, **kwargs) -> AcousticEncodingResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode_session(self, *args, **kwargs) -> AcousticEncodingSession:  # noqa: ANN002, ANN003
    ...
