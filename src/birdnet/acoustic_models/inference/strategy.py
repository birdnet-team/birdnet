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
from birdnet.acoustic_models.inference.states import (
  FilesAnalyzerResources,
  LoggingResources,
  ProcessingResources,
  ProducerResources,
  RingBufferResources,
  StatisticsResources,
  WorkerResources,
)
from birdnet.backends import InferenceBackendLoader


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
    logging_resources: LoggingResources,
    ring_buffer_resources: RingBufferResources,
    producer_resources: ProducerResources,
    processing_state: ProcessingResources,
    stats_resources: StatisticsResources,
    worker_resources: WorkerResources,
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
    pred_result: ResultType,
    file_durations: np.ndarray,
    memory_layout: RingBufferResources,
    analyzer_resources: FilesAnalyzerResources,
    stats_resources: StatisticsResources,
  ) -> MinimalBenchmarkMetaBase: ...

  @abstractmethod
  def create_full_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    pred_result: ResultType,
    file_durations: np.ndarray,
    memory_layout: RingBufferResources,
    perf_result: PerformanceTrackingResult,
    analyzer_resources: FilesAnalyzerResources,
    stats_resources: StatisticsResources,
  ) -> FullBenchmarkMetaBase: ...

  @abstractmethod
  def get_benchmark_dir_name(self) -> str: ...

  @abstractmethod
  def save_results_extra(
    self, result: ResultType, benchmark_run_out_dir: Path, iso_time: str
  ) -> list[Path]: ...


def get_file_formats(file_paths: OrderedSet[Path]) -> str:
  return ", ".join(sorted({x.suffix[1:].upper() for x in file_paths}))
