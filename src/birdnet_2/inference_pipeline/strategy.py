from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
)
from birdnet.acoustic_models.inference.worker import WorkerBase


class PredictionStrategy(Generic[ResultType, ConfigType, TensorType], ABC):
  # def __init__(self, config: PredictionConfig, specific_config: ConfigType) -> None:
  #   self._config = config
  #   self._specific_config = specific_config

  @abstractmethod
  def validate_config(
    self, config: PredictionConfig, specific_config: ConfigType
  ) -> None: ...

  @abstractmethod
  def create_tensor(
    self,
    config: PredictionConfig,
    specific_config: ConfigType,
    resources: PipelineResources,
  ) -> TensorType: ...

  @abstractmethod
  def create_workers(
    self,
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
    self, result: ResultType, benchmark_run_out_dir: Path, iso_time: str
  ) -> list[Path]: ...


def get_file_formats(file_paths: OrderedSet[Path]) -> str:
  return ", ".join(sorted({x.suffix[1:].upper() for x in file_paths}))
