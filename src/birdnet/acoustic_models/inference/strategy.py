from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Generic

import numpy as np
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic_models.inference.perf_tracker import PerformanceTrackingResult
from birdnet.acoustic_models.inference.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference.states import (
  MemoryLayout,
  ProcessingState,
  SharedResources,
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
    memory_layout: MemoryLayout,
  ) -> TensorType: ...

  @abstractmethod
  def create_workers(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    devices: list[str],
    backend_loader: InferenceBackendLoader,
    shared_resources: SharedResources,
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
    processing_state: ProcessingState,
    start_timepoint: datetime,
    end_timepoint: datetime,
    wall_time_s: float,
    file_durations: np.ndarray,
    memory_layout: MemoryLayout,
  ) -> MinimalBenchmarkMetaBase: ...

  @abstractmethod
  def create_full_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    pred_result: ResultType,
    processing_state: ProcessingState,
    start_time: float,
    start_timepoint: datetime,
    end_timepoint: datetime,
    wall_time_s: float,
    file_durations: np.ndarray,
    memory_layout: MemoryLayout,
    perf_result: PerformanceTrackingResult,
  ) -> FullBenchmarkMetaBase: ...

  @abstractmethod
  def get_benchmark_dir_name(self) -> str: ...

  @abstractmethod
  def save_results(self, result: ResultType, npz_path: Path, csv_path: Path) -> str: ...


def get_file_formats(file_paths: OrderedSet[Path]) -> str:
  """Extrahiert Dateiformate aus Pfaden"""
  return ", ".join(sorted({x.suffix[1:].upper() for x in file_paths}))
