from __future__ import annotations

import multiprocessing
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.base import PredictionResultBase
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
)
from birdnet.helper import (
  SF_FORMATS,
  get_supported_audio_files,
  is_supported_audio_file,
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

  @property
  def segment_size_samples(self) -> int:
    return int(self.segment_size_s * self.sample_rate)

  @property
  def n_species(self) -> int:
    return len(self.species_list)


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

  @property
  def result_dtype(self) -> DTypeLike:
    result_dtype = np.float16 if self.half_precision else np.float32
    return result_dtype

  @property
  def n_slots(self) -> int:
    n_slots = self.workers + (self.workers * self.prefetch_ratio)
    return n_slots

  @classmethod
  def validate_feeders(cls, feeders: Any) -> int:  # noqa: ANN401
    if not isinstance(feeders, int):
      raise TypeError("feeders must be an integer")
    if not feeders >= 1:
      raise ValueError("feeders must be >= 1")
    max_threads = multiprocessing.cpu_count() or 1
    if not feeders <= max_threads:
      raise ValueError(f"feeders must be <= {max_threads}")
    return feeders

  @classmethod
  def validate_workers(cls, workers: Any) -> int:  # noqa: ANN401
    if not isinstance(workers, int):
      raise TypeError("workers must be an integer")
    if not workers >= 1:
      raise ValueError("workers must be >= 1")
    max_threads = multiprocessing.cpu_count() or 1
    if not workers <= max_threads:
      raise ValueError(f"workers must be <= {max_threads}")
    return workers

  @classmethod
  def validate_batch_size(cls, batch_size: Any) -> int:  # noqa: ANN401
    if not isinstance(batch_size, int):
      raise TypeError("batch size must be an integer")
    if not batch_size >= 1:
      raise ValueError("batch size must be >= 1")
    return batch_size

  @classmethod
  def validate_prefetch_ratio(cls, prefetch_ratio: Any) -> int:  # noqa: ANN401
    if not isinstance(prefetch_ratio, int):
      raise TypeError("prefetch ratio must be an integer")
    if not prefetch_ratio >= 0:
      raise ValueError("prefetch ratio must be >= 0")
    return prefetch_ratio

  @classmethod
  def validate_overlap_duration(
    cls,
    overlap_duration_s: Any,  # noqa: ANN401
    segment_size_s: float,
  ) -> float:
    assert segment_size_s > 0
    if not isinstance(overlap_duration_s, int | float):
      raise TypeError("overlap duration must be a number")
    if not 0 <= overlap_duration_s < segment_size_s:
      raise ValueError(f"overlap duration must be in [0, {segment_size_s})")
    return overlap_duration_s

  @classmethod
  def validate_device(
    cls,
    device: Any | list[Any],  # noqa: ANN401
    workers: int,
  ) -> str | list[str]:
    if isinstance(device, str):
      if "GPU" not in device and "CPU" not in device:
        raise ValueError("device name must contain 'CPU' or 'GPU'")
    elif isinstance(device, list):
      if len(device) != workers:
        raise ValueError(
          f"device list length ({len(device)}) must match workers count ({workers})"
        )
      for d in device:
        cls.validate_device(d, workers)
    else:
      raise TypeError("device must be a string or a list of strings")
    return device

  @classmethod
  def validate_half_precision(cls, half_precision: Any) -> bool:  # noqa: ANN401
    if not isinstance(half_precision, bool):
      raise TypeError("half precision must be a boolean")
    return half_precision

  @classmethod
  def validate_max_audio_duration_min(cls, max_audio_duration_min: Any) -> float:  # noqa: ANN401
    if not isinstance(max_audio_duration_min, int | float):
      raise TypeError("maximum audio duration must be a number")
    if not max_audio_duration_min > 0:
      raise ValueError("maximum audio duration must be > 0")
    return max_audio_duration_min


@dataclass(frozen=True)
class FilteringConfig:
  bandpass_fmin: int
  bandpass_fmax: int

  @classmethod
  def validate_bandpass_frequencies(
    cls,
    bandpass_fmin: Any,  # noqa: ANN401
    bandpass_fmax: Any,  # noqa: ANN401
    supported_fmin: int,
    supported_fmax: int,
  ) -> tuple[int, int]:
    if bandpass_fmin is None:
      raise ValueError("bandpass minimum frequence must be specified")
    if bandpass_fmax is None:
      raise ValueError("bandpass maximum frequence must be specified")

    if not isinstance(bandpass_fmin, int) or not isinstance(bandpass_fmax, int):
      raise TypeError("bandpass frequencies must be integers")

    if not supported_fmin <= bandpass_fmin < bandpass_fmax <= supported_fmax:
      raise ValueError(
        f"bandpass frequencies must be in the range [{supported_fmin}, {supported_fmax}] and fmin < fmax"
      )
    return bandpass_fmin, bandpass_fmax


@dataclass(frozen=True)
class OutputConfig:
  show_stats: None | Literal["minimal", "progress", "benchmark"]

  @classmethod
  def validate_show_stats(
    cls,
    show_stats: Any,  # noqa: ANN401
  ) -> Literal["minimal", "progress", "benchmark"]:
    if show_stats is not None and show_stats not in (
      "minimal",
      "progress",
      "benchmark",
    ):
      raise ValueError("show stats must be one of 'minimal', 'progress' or 'benchmark'")
    return show_stats


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
  custom_species_list: set[str] | None
  apply_sigmoid: bool
  sigmoid_sensitivity: float | None

  @classmethod
  def validate_top_k(
    cls,
    top_k: Any,  # noqa: ANN401
    max_value: int,
  ) -> int:
    if not isinstance(top_k, int):
      raise TypeError("top k must be an integer")
    if not 1 <= top_k <= max_value:
      raise ValueError(f"top k must be in the range [1, {max_value}]")
    return top_k

  @classmethod
  def validate_default_confidence_threshold(
    cls,
    default_confidence_threshold: Any,  # noqa: ANN401
  ) -> float:
    if not isinstance(default_confidence_threshold, int | float):
      raise TypeError("default confidence threshold must be a number")
    return default_confidence_threshold

  @classmethod
  def validate_custom_confidence_thresholds(
    cls,
    custom_confidence_thresholds: Any,  # noqa: ANN401
    model_species: Collection[str],
  ) -> dict[str, float]:
    if not isinstance(custom_confidence_thresholds, dict):
      raise TypeError("custom confidence thresholds must be a dictionary")
    for k, v in custom_confidence_thresholds.items():
      if not isinstance(k, str):
        raise TypeError("custom confidence threshold keys must be strings")
      if k not in model_species:
        raise ValueError(f"species '{k}' is not available in the model")
      if not isinstance(v, int | float):
        raise TypeError("custom confidence threshold values must be numbers")
    return custom_confidence_thresholds

  @classmethod
  def validate_custom_species_list(
    cls,
    custom_species_list: Any,  # noqa: ANN401
    model_species: Collection[str],
  ) -> set[str]:
    if not isinstance(custom_species_list, Collection):
      raise TypeError(
        "custom species list must be a collection (list, set, tuple, etc.)"
      )
    for species in custom_species_list:
      if not isinstance(species, str):
        raise TypeError("custom species list must contain strings")
      if species not in model_species:
        raise ValueError(f"species '{species}' is not available in the model")
    return set(custom_species_list)

  @classmethod
  def validate_sigmoid_sensitivity(
    cls,
    sigmoid_sensitivity: Any,  # noqa: ANN401
  ) -> float:
    if not isinstance(sigmoid_sensitivity, int | float):
      raise TypeError("sigmoid sensitivity must be a number")
    if not 0.5 <= sigmoid_sensitivity <= 1.5:
      raise ValueError("sigmoid sensitivity must be in the range [0.5, 1.5]")
    return sigmoid_sensitivity


@dataclass(frozen=True)
class PredictionConfig:
  input_files: set[Path]
  model_conf: ModelConfig
  processing_conf: ProcessingConfig
  filtering_conf: FilteringConfig
  output_conf: OutputConfig

  @classmethod
  def validate_input_files(
    cls,
    input_files: Any | Iterable[Any],  # noqa: ANN401
  ) -> set[Path]:
    parsed_audio_paths: set[Path] = set()

    if isinstance(input_files, Path | str):
      input_files = (Path(input_files),)

    if isinstance(input_files, Iterable):
      for inp_audio in input_files:
        if isinstance(inp_audio, Path | str):
          inp_path = Path(inp_audio)
          if inp_path.is_file():
            if is_supported_audio_file(inp_path):
              parsed_audio_paths.add(inp_path.absolute())
            else:
              raise ValueError(
                f"Input file '{inp_path}' is not a supported audio format! Supported formats: {sorted(SF_FORMATS)}."
              )
          elif inp_path.is_dir():
            parsed_audio_paths.update(get_supported_audio_files(inp_path))
          else:
            raise ValueError(f"Input path '{inp_path}' was not found.")
        else:
          raise ValueError(f"Unsupported input type: {type(inp_audio)}")
    else:
      raise ValueError(f"Unsupported input type: {type(input_files)}")

    for p in parsed_audio_paths:
      assert p.is_absolute()
    return parsed_audio_paths
