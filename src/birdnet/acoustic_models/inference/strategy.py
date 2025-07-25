from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

import numpy as np
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic_models.inference.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference.perf_tracker import PerformanceTrackingResult
from birdnet.acoustic_models.inference.resources import (
  FilesAnalyzerResources,
  PipelineResources,
  RingBufferResources,
)


class PredictionStrategy(Generic[ResultType, ConfigType, TensorType], ABC):
  @abstractmethod
  def validate_config(
    self, config: PredictionConfig, specific_config: ConfigType
  ) -> None: ...

  @abstractmethod
  def create_tensor(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    memory_layout: RingBufferResources,
    analyzer_resources: FilesAnalyzerResources,
  ) -> TensorType: ...

  @abstractmethod
  def create_workers(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
  ) -> list[mp.Process]: ...

  @abstractmethod
  def create_result(
    self,
    tensor: TensorType,
    config: PredictionConfig,
    file_paths: OrderedSet[Path],
    file_durations: np.ndarray,
  ) -> ResultType: ...

  @abstractmethod
  def create_minimal_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    pred_result: ResultType,
    file_durations: np.ndarray,
  ) -> MinimalBenchmarkMetaBase: ...

  @abstractmethod
  def create_full_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    pred_result: ResultType,
    file_durations: np.ndarray,
    perf_result: PerformanceTrackingResult,
  ) -> FullBenchmarkMetaBase: ...

  @abstractmethod
  def get_benchmark_dir_name(self) -> str: ...

  @abstractmethod
  def save_results_extra(
    self, result: ResultType, benchmark_run_out_dir: Path, iso_time: str
  ) -> list[Path]: ...


def get_file_formats(file_paths: OrderedSet[Path]) -> str:
  return ", ".join(sorted({x.suffix[1:].upper() for x in file_paths}))
