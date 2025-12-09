from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
)


class InferenceStrategyBase(Generic[ResultType, ConfigType, TensorType], ABC):
  @abstractmethod
  def validate_config(
    self, config: InferenceConfig, specific_config: ConfigType
  ) -> None: ...

  @abstractmethod
  def create_tensor(
    self,
    session_id: str,
    config: InferenceConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    n_inputs: int,
  ) -> TensorType: ...

  @abstractmethod
  def create_workers(
    self,
    session_id: str,
    config: InferenceConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
  ) -> list[WorkerBase]: ...

  @abstractmethod
  def create_files_result(
    self,
    tensor: TensorType,
    config: InferenceConfig,
    resources: PipelineResources,
    files: list[Path],
  ) -> ResultType: ...

  @abstractmethod
  def create_array_result(
    self,
    tensor: TensorType,
    config: InferenceConfig,
    resources: PipelineResources,
  ) -> ResultType: ...

  @abstractmethod
  def create_minimal_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
    pred_result: ResultType,
  ) -> MinimalBenchmarkMetaBase: ...

  @abstractmethod
  def create_full_benchmark_meta(
    self,
    config: InferenceConfig,
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
