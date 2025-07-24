from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.base import PredictionResultBase
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
)

ResultType = TypeVar("ResultType", bound="PredictionResultBase")
ConfigType = TypeVar("ConfigType", bound="SpecificConfigBase")
TensorType = TypeVar("TensorType", bound="TensorBase")


@dataclass(frozen=True)
class ModelConfig:
  species_list: OrderedSet[str]
  path: Path
  backend: MODEL_BACKENDS
  backend_kwargs: dict
  version: ACOUSTIC_MODEL_VERSIONS
  segment_size_s: float
  sample_rate: int
  sig_fmin: int
  sig_fmax: int
  precision: MODEL_PRECISIONS
  is_custom: bool


@dataclass(frozen=True)
class ProcessingConfig:
  feeders: int
  workers: int
  batch_size: int
  prefetch_ratio: int
  overlap_duration_s: float
  half_precision: bool
  max_audio_duration_min: float | None
  device: str | list[str]


@dataclass(frozen=True)
class FilteringConfig:
  use_bandpass: bool
  bandpass_fmin: int | None
  bandpass_fmax: int | None


@dataclass(frozen=True)
class OutputConfig:
  show_stats: Literal["no", "minimal", "progress", "benchmark"]


@dataclass(frozen=True)
class SpecificConfigBase:
  pass


@dataclass(frozen=True)
class EmbeddingsConfig(SpecificConfigBase):
  emb_dim: int


@dataclass(frozen=True)
class ScoresConfig(SpecificConfigBase):
  top_k: int | None
  default_confidence_threshold: float | None
  custom_confidence_thresholds: dict[str, float] | None
  apply_sigmoid: bool
  sigmoid_sensitivity: float | None
  custom_species_list: set[str] | None


@dataclass(frozen=True)
class PredictionConfig:
  input_files: Path | str | Iterable[Path | str]
  model_conf: ModelConfig
  processing_conf: ProcessingConfig
  filtering_conf: FilteringConfig
  output_conf: OutputConfig


def validate_common_config(config: PredictionConfig) -> None:
  """Validiert gemeinsame Parameter"""
  if not config.processing_conf.batch_size >= 1:
    raise ValueError("batch_size must be >= 1")
  if not config.processing_conf.feeders >= 1:
    raise ValueError("feeders must be >= 1")
  if not config.processing_conf.workers >= 1:
    raise ValueError("workers must be >= 1")
  if not config.processing_conf.prefetch_ratio >= 0:
    raise ValueError("prefetch_ratio must be >= 0")
  if (
    not 0
    <= config.processing_conf.overlap_duration_s
    < config.model_conf.segment_size_s
  ):
    raise ValueError(
      f"overlap_duration_s must be in [0, {config.model_conf.segment_size_s})"
    )

  if config.filtering_conf.use_bandpass:
    if (
      config.filtering_conf.bandpass_fmin is None
      or config.filtering_conf.bandpass_fmax is None
    ):
      raise ValueError("bandpass frequencies required when use_bandpass=True")
    if (
      config.filtering_conf.bandpass_fmin < 0
      or config.filtering_conf.bandpass_fmax <= config.filtering_conf.bandpass_fmin
    ):
      raise ValueError("invalid bandpass frequency range")

  if (
    config.processing_conf.max_audio_duration_min is not None
    and not config.processing_conf.max_audio_duration_min > 0
  ):
    raise ValueError("max_audio_duration_min must be None or > 0")

  if (
    isinstance(config.processing_conf.device, list)
    and len(config.processing_conf.device) != config.processing_conf.workers
  ):
    raise ValueError(
      f"device list length must match workers count ({config.processing_conf.workers})"
    )

  devices = (
    config.processing_conf.device
    if isinstance(config.processing_conf.device, list)
    else [config.processing_conf.device] * config.processing_conf.workers
  )
  if config.model_conf.backend == MODEL_BACKEND_TF:
    for d in devices:
      if "GPU" in d:
        raise ValueError("GPU not supported for TFLite backend")
