from __future__ import annotations

from pathlib import Path

import numpy as np

from birdnet.acoustic.inference.configs import (
  EncodingConfig,
  InferenceConfig,
)
from birdnet.acoustic.inference.core.encoding.encoding_benchmarking import (
  FullBenchmarkEmbMeta,
  MinimalBenchmarkEmbMeta,
)
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticDataEncodingResult,
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
  AcousticEncodingTensor,
)
from birdnet.acoustic.inference.core.encoding.encoding_worker import EncodingWorker
from birdnet.acoustic.inference.core.worker import WorkerBase
from birdnet.acoustic.inference.resources import (
  PipelineResources,
)
from birdnet.acoustic.inference.strategy import (
  InferenceStrategyBase,
)


class EncodingStrategy(
  InferenceStrategyBase[
    AcousticEncodingResultBase, EncodingConfig, AcousticEncodingTensor
  ]
):
  def create_tensor(
    self,
    session_id: str,
    config: InferenceConfig,
    specific_config: EncodingConfig,
    resources: PipelineResources,
    n_inputs: int,
  ) -> AcousticEncodingTensor:
    return AcousticEncodingTensor(
      session_id,
      n_inputs,
      emb_dim=specific_config.emb_dim,
      half_precision=config.processing_conf.half_precision,
      segment_indices_dtype=resources.ring_buffer_resources.rf_segment_indices.dtype,
      input_indices_dtype=resources.ring_buffer_resources.rf_file_indices.dtype,
      max_segment_index=resources.analyzer_resources.max_segment_idx_ptr,
    )

  def create_workers(
    self,
    session_id: str,
    config: InferenceConfig,
    specific_config: EncodingConfig,
    resources: PipelineResources,
  ) -> list[WorkerBase]:
    return [
      EncodingWorker(**self._worker_kwargs(session_id, config, resources, i))
      for i in range(config.processing_conf.workers)
    ]

  def create_files_result(
    self,
    tensor: AcousticEncodingTensor,
    config: InferenceConfig,
    resources: PipelineResources,
    files: list[Path],
  ) -> AcousticEncodingResultBase:
    return AcousticFileEncodingResult(
      **self._result_kwargs(config),
      tensor=tensor,
      files=files,
      file_durations=resources.analyzer_resources.input_durations,
    )

  def build_single_file_result(
    self,
    config: InferenceConfig,
    file_path: Path,
    arrays: tuple[object, ...],
    is_invalid: bool,
    duration_s: float,
  ) -> AcousticFileEncodingResult:
    from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
      PrebuiltEncodingTensor,
    )

    emb, emb_masked = arrays
    tensor = PrebuiltEncodingTensor(
      emb=emb,  # type: ignore[arg-type]
      emb_masked=emb_masked,  # type: ignore[arg-type]
    )
    tensor.set_unprocessable_inputs({0} if is_invalid else set())

    return AcousticFileEncodingResult(
      **self._result_kwargs(config),
      tensor=tensor,
      files=[Path(file_path)],
      file_durations=np.array([duration_s], dtype=np.float32),
    )

  def create_array_result(
    self,
    tensor: AcousticEncodingTensor,
    config: InferenceConfig,
    resources: PipelineResources,
  ) -> AcousticEncodingResultBase:
    return AcousticDataEncodingResult(
      **self._result_kwargs(config),
      tensor=tensor,
      input_durations=resources.analyzer_resources.input_durations,
    )

  def create_minimal_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: EncodingConfig,
    resources: PipelineResources,
    pred_result: AcousticEncodingResultBase,
  ) -> MinimalBenchmarkEmbMeta:
    return MinimalBenchmarkEmbMeta(
      **self._minimal_meta_kwargs(config, resources, pred_result)
    )

  def create_full_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: EncodingConfig,
    resources: PipelineResources,
    pred_result: AcousticEncodingResultBase,
  ) -> FullBenchmarkEmbMeta:
    return FullBenchmarkEmbMeta(
      **self._full_meta_kwargs(config, resources, pred_result),
      model_emb_dim=specific_config.emb_dim,
    )

  def get_benchmark_dir_name(self) -> str:
    return "encoding"

  def save_results_extra(
    self, result: AcousticEncodingResultBase, benchmark_run_out_dir: Path, prepend: str
  ) -> list[Path]:
    return []
