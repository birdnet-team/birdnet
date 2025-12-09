from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ordered_set import OrderedSet

from birdnet.backends import VersionedBackendProtocol
from birdnet.globals import GEO_MODEL_VERSIONS


@dataclass(frozen=True)
class ModelConfig:
  species_list: OrderedSet[str]
  path: Path
  version: GEO_MODEL_VERSIONS
  is_custom: bool
  backend_type: type[VersionedBackendProtocol]
  backend_kwargs: dict[str, Any]

  @property
  def n_species(self) -> int:
    return len(self.species_list)


@dataclass(frozen=True)
class ProcessingConfig:
  half_precision: bool
  device: str

  @classmethod
  def validate_device(
    cls,
    device: Any,  # noqa: ANN401
  ) -> str:
    if isinstance(device, str):
      if "GPU" not in device and "CPU" not in device:
        raise ValueError("device name must contain 'CPU' or 'GPU'")
    else:
      raise TypeError("device must be a string")
    return device

  @classmethod
  def validate_half_precision(cls, half_precision: Any) -> bool:  # noqa: ANN401
    if not isinstance(half_precision, bool):
      raise TypeError("half precision must be a boolean")
    return half_precision


@dataclass(frozen=True)
class PredictionConfig:
  min_confidence: float

  @classmethod
  def validate_min_confidence(cls, min_confidence: Any) -> float:  # noqa: ANN401
    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0)."
      )
    return float(min_confidence)


@dataclass(frozen=True)
class RunConfig:
  latitude: float
  longitude: float
  week: int

  @classmethod
  def validate_latitude(cls, latitude: Any) -> float:  # noqa: ANN401
    if not isinstance(latitude, float | int):
      raise TypeError("latitude must be a float")
    if not -90 <= latitude <= 90:
      raise ValueError(
        "Value for 'latitude' is invalid! It needs to be in interval [-90, 90]."
      )
    return float(latitude)

  @classmethod
  def validate_longitude(cls, longitude: Any) -> float:  # noqa: ANN401
    if not isinstance(longitude, float | int):
      raise TypeError("longitude must be a float")
    if not -180 <= longitude <= 180:
      raise ValueError(
        "Value for 'longitude' is invalid! It needs to be in interval [-180, 180]."
      )
    return float(longitude)

  @classmethod
  def validate_week(cls, week: Any) -> int:  # noqa: ANN401
    if week is None:
      return -1

    if not isinstance(week, int):
      raise TypeError("Value for 'week' is invalid! It must be an integer.")

    if week is not None and not (1 <= week <= 48):
      raise ValueError(
        "Value for 'week' is invalid! It needs to be either None or in interval [1, 48]."
      )

    return week


@dataclass(frozen=True)
class InferenceConfig:
  model_conf: ModelConfig
  processing_conf: ProcessingConfig
