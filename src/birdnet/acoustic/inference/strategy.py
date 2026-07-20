from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

from birdnet.acoustic.inference.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic.inference.core.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic.inference.core.worker import WorkerBase
from birdnet.acoustic.inference.resources import (
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

  def build_single_file_result(
    self,
    config: InferenceConfig,
    file_path: Path,
    arrays: tuple[object, ...],
    is_invalid: bool,
    duration_s: float,
  ) -> ResultType:
    """Build a single-file result from already-materialised per-file arrays.

    ``arrays`` is the strategy-specific tuple produced by the tensor's
    ``copy_file_slice`` (predictions: species ids/probs/masked; encodings:
    embeddings/mask). Used by the per-file completion dispatcher
    (``on_file_complete``).
    """
    raise NotImplementedError(
      "This inference strategy does not support per-file completion results."
    )

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
