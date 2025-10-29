from __future__ import annotations

import multiprocessing
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import psutil
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.backends import (
  VersionedBackendProtocol,
)
from birdnet.base import PredictionResultBase
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
)
from birdnet.helper import (
  SF_FORMATS,
  get_supported_audio_files,
  is_supported_audio_file,
  validate_species_list,
)

ResultType = TypeVar("ResultType", bound="PredictionResultBase")
ConfigType = TypeVar("ConfigType", bound="SpecificConfigBase")
TensorType = TypeVar("TensorType", bound="TensorBase")


@dataclass(frozen=True)
class ModelConfig:
  species_list: OrderedSet[str]
  path: Path
  version: ACOUSTIC_MODEL_VERSIONS
  segment_size_s: float
  sample_rate: int
  sig_fmin: int
  sig_fmax: int
  is_custom: bool
  backend_type: type[VersionedBackendProtocol]
  backend_kwargs: dict[str, Any]

  @property
  def segment_size_samples(self) -> int:
    return int(self.segment_size_s * self.sample_rate)

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @classmethod
  def validate_backend_supports_embeddings(
    cls, backend: type[VersionedBackendProtocol]
  ) -> None:
    if not backend.emb_supported():
      raise ValueError("loaded backend does not support embeddings")


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
  max_n_files: int

  @property
  def result_dtype(self) -> DTypeLike:
    result_dtype = np.float16 if self.half_precision else np.float32
    return result_dtype

  @property
  def n_slots(self) -> int:
    n_slots = self.workers + (self.workers * self.prefetch_ratio)
    return n_slots

  @classmethod
  def validate_max_n_files(cls, max_n_files: Any) -> int:  # noqa: ANN401
    if not isinstance(max_n_files, int):
      raise TypeError("max_n_files must be an integer")
    if not max_n_files >= 1:
      raise ValueError("max_n_files must be >= 1")
    if not max_n_files <= 2**64:
      raise ValueError("max_n_files must be <= 2^64")
    return max_n_files

  @classmethod
  def validate_n_feeders(cls, n_feeders: Any) -> int:  # noqa: ANN401
    if not isinstance(n_feeders, int):
      raise TypeError("n_feeders must be an integer")
    if not n_feeders >= 1:
      raise ValueError("n_feeders must be >= 1")
    max_threads = multiprocessing.cpu_count() or 1
    if not n_feeders <= max_threads:
      raise ValueError(f"n_feeders must be <= {max_threads}")
    return n_feeders

  @classmethod
  def validate_n_workers(cls, n_workers: Any) -> int:  # noqa: ANN401
    if n_workers is None:
      n_physical_cores = psutil.cpu_count(logical=False) or 1
      n_workers = n_physical_cores
      return n_workers
    if not isinstance(n_workers, int):
      raise TypeError("n_workers must be an integer")
    if not n_workers >= 1:
      raise ValueError("n_workers must be >= 1")
    max_threads = psutil.cpu_count(logical=True) or 1

    if not n_workers <= max_threads:
      raise ValueError(f"n_workers must be <= {max_threads}")
    return n_workers

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
      raise TypeError("prefetch_ratio must be an integer")
    if not prefetch_ratio >= 0:
      raise ValueError("prefetch_ratio must be >= 0")
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
        raise ValueError("device must contain 'CPU' or 'GPU'")
    elif isinstance(device, list):
      if len(device) != workers:
        raise ValueError(
          f"length of device list ({len(device)}) must match number of "
          f"workers ({workers})"
        )
      for d in device:
        cls.validate_device(d, workers)
    else:
      raise TypeError("device must be a string or a list of strings")
    return device

  @classmethod
  def validate_half_precision(cls, half_precision: Any) -> bool:  # noqa: ANN401
    if not isinstance(half_precision, bool):
      raise TypeError("half_precision must be a boolean")
    return half_precision

  @classmethod
  def validate_max_audio_duration_min(cls, max_audio_duration_min: Any) -> float:  # noqa: ANN401
    if not isinstance(max_audio_duration_min, int | float):
      raise TypeError("max_audio_duration_min must be a number")
    if not max_audio_duration_min > 0:
      raise ValueError("max_audio_duration_min must be > 0")
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
    if isinstance(custom_species_list, Path | str):
      custom_species_list_path = Path(custom_species_list)
      custom_species_list = validate_species_list(custom_species_list_path)
    elif not isinstance(custom_species_list, Collection):
      raise TypeError(
        "custom species list must be a str, path or collection (list, set, tuple, etc.)"
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

    if len(parsed_audio_paths) == 0:
      raise ValueError("No valid audio files were found in the provided input paths.")

    return parsed_audio_paths
