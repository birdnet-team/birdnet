from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
)


class PredictionStrategy(Generic[ResultType, ConfigType, TensorType], ABC):
  @abstractmethod
  def validate_config(
    self, config: PredictionConfig, specific_config: ConfigType
  ) -> None: ...

  @abstractmethod
  def create_tensor(
    self,
    session_id: str,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    n_files: int,
  ) -> TensorType: ...

  @abstractmethod
  def create_workers(
    self,
    session_id: str,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
  ) -> list[WorkerBase]: ...

  @abstractmethod
  def create_result(
    self,
    tensor: TensorType,
    config: PredictionConfig,
    resources: PipelineResources,
    files: OrderedSet[Path],
  ) -> ResultType: ...

  @abstractmethod
  def create_minimal_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    pred_result: ResultType,
  ) -> MinimalBenchmarkMetaBase: ...

  @abstractmethod
  def create_full_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    pred_result: ResultType,
  ) -> FullBenchmarkMetaBase: ...

  @abstractmethod
  def get_benchmark_dir_name(self) -> str: ...

  @abstractmethod
  def save_results_extra(
    self, result: ResultType, benchmark_run_out_dir: Path, prepend: str
  ) -> list[Path]: ...
