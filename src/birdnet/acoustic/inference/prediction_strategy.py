from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from birdnet.acoustic.inference.configs import (
  InferenceConfig,
  PredictionConfig,
)
from birdnet.acoustic.inference.core.prediction.prediction_benchmarking import (
  FullBenchmarkMeta,
  MinimalBenchmarkMeta,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
  AcousticPredictionResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_tensor import (
  AcousticPredictionTensor,
)
from birdnet.acoustic.inference.core.prediction.prediction_worker import (
  PredictionWorker,
)
from birdnet.acoustic.inference.core.worker import WorkerBase
from birdnet.acoustic.inference.resources import (
  PipelineResources,
)
from birdnet.acoustic.inference.strategy import (
  InferenceStrategyBase,
)


class PredictionStrategy(
  InferenceStrategyBase[
    AcousticPredictionResultBase, PredictionConfig, AcousticPredictionTensor
  ]
):
  def create_tensor(
    self,
    session_id: str,
    config: InferenceConfig,
    specific_config: PredictionConfig,
    resources: PipelineResources,
    n_inputs: int,
  ) -> AcousticPredictionTensor:
    return AcousticPredictionTensor(
      session_id,
      n_inputs,
      top_k=self.get_top_k(config, specific_config),
      n_species=config.model_conf.n_species,
      half_precision=config.processing_conf.half_precision,
      segment_indices_dtype=resources.ring_buffer_resources.rf_segment_indices.dtype,
      input_indices_dtype=resources.ring_buffer_resources.rf_file_indices.dtype,
      max_segment_index=resources.analyzer_resources.max_segment_idx_ptr,
    )

  def get_top_k(
    self, config: InferenceConfig, specific_config: PredictionConfig
  ) -> int:
    return (
      specific_config.top_k
      if specific_config.top_k is not None
      else config.model_conf.n_species
    )

  def create_workers(
    self,
    session_id: str,
    config: InferenceConfig,
    specific_config: PredictionConfig,
    resources: PipelineResources,
  ) -> list[WorkerBase]:
    species_blacklist = create_species_blacklist(config, specific_config)
    species_thresholds = create_thresholds(config, specific_config)
    top_k = self.get_top_k(config, specific_config)

    return [
      PredictionWorker(
        **self._worker_kwargs(session_id, config, resources, i),
        top_k=top_k,
        species_thresholds=species_thresholds,
        species_blacklist=species_blacklist,
        apply_sigmoid=specific_config.apply_sigmoid,
        apply_softmax=specific_config.apply_softmax,
        sigmoid_sensitivity=specific_config.sigmoid_sensitivity,
        model_applies_sigmoid=config.model_conf.backend_type.applies_sigmoid(),
      )
      for i in range(config.processing_conf.workers)
    ]

  def create_files_result(
    self,
    tensor: AcousticPredictionTensor,
    config: InferenceConfig,
    resources: PipelineResources,
    files: list[Path],
  ) -> AcousticPredictionResultBase:
    return AcousticFilePredictionResult(
      **self._result_kwargs(config),
      tensor=tensor,
      files=files,
      species_list=config.model_conf.species_list,
      file_durations=resources.analyzer_resources.input_durations,
    )

  def _species_list_array(self, config: InferenceConfig) -> npt.NDArray:
    # Built once per strategy (i.e. per session) and reused for every per-file
    # result, so the completion callback does not pay to rebuild the (large)
    # species-name array for each file.
    cached = getattr(self, "_cached_species_array", None)
    if cached is None:
      species_list = config.model_conf.species_list
      max_len = max(map(len, species_list))
      cached = np.array(list(species_list), dtype=f"<U{max_len}")
      cached.setflags(write=False)
      self._cached_species_array = cached
    return cached

  def build_single_file_result(
    self,
    config: InferenceConfig,
    file_path: Path,
    arrays: tuple[object, ...],
    is_invalid: bool,
    duration_s: float,
  ) -> AcousticFilePredictionResult:
    from birdnet.acoustic.inference.core.prediction.prediction_tensor import (
      PrebuiltPredictionTensor,
    )

    species_ids, species_probs, species_masked = arrays
    tensor = PrebuiltPredictionTensor(
      species_ids=species_ids,  # type: ignore[arg-type]
      species_probs=species_probs,  # type: ignore[arg-type]
      species_masked=species_masked,  # type: ignore[arg-type]
    )
    tensor.set_unprocessable_inputs({0} if is_invalid else set())

    return AcousticFilePredictionResult(
      **self._result_kwargs(config),
      tensor=tensor,
      files=[Path(file_path)],
      species_list=config.model_conf.species_list,
      species_list_array=self._species_list_array(config),
      file_durations=np.array([duration_s], dtype=np.float32),
    )

  def create_array_result(
    self,
    tensor: AcousticPredictionTensor,
    config: InferenceConfig,
    resources: PipelineResources,
  ) -> AcousticPredictionResultBase:
    return AcousticDataPredictionResult(
      **self._result_kwargs(config),
      tensor=tensor,
      species_list=config.model_conf.species_list,
      input_durations=resources.analyzer_resources.input_durations,
    )

  def create_minimal_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: PredictionConfig,
    resources: PipelineResources,
    pred_result: AcousticPredictionResultBase,
  ) -> MinimalBenchmarkMeta:
    return MinimalBenchmarkMeta(
      **self._minimal_meta_kwargs(config, resources, pred_result)
    )

  def create_full_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: PredictionConfig,
    resources: PipelineResources,
    pred_result: AcousticPredictionResultBase,
  ) -> FullBenchmarkMeta:
    return FullBenchmarkMeta(
      **self._full_meta_kwargs(config, resources, pred_result),
      param_top_k=specific_config.top_k,
      param_sigmoid_apply=specific_config.apply_sigmoid,
      param_softmax_apply=specific_config.apply_softmax,
      param_sigmoid_sensitivity=specific_config.sigmoid_sensitivity
      if specific_config.apply_sigmoid
      else None,
      param_confidence_threshold_default=specific_config.default_confidence_threshold,
      param_custom_species=len(specific_config.custom_species_list)
      if specific_config.custom_species_list
      else 0,
      param_confidence_threshold_custom=len(
        specific_config.custom_confidence_thresholds
      )
      if specific_config.custom_confidence_thresholds
      else 0,
    )

  def get_benchmark_dir_name(self) -> str:
    return "predictions"

  def save_results_extra(
    self,
    result: AcousticPredictionResultBase,
    benchmark_run_out_dir: Path,
    prepend: str,
  ) -> list[Path]:
    print("Saving result using CSV format (.csv)...")
    csv_path = benchmark_run_out_dir / f"{prepend}-result.csv"
    result.to_csv(csv_path, encoding="utf-8", silent=False)
    return [csv_path]


def create_thresholds(
  config: InferenceConfig, pred_conf: PredictionConfig
) -> npt.NDArray:
  default_threshold = pred_conf.default_confidence_threshold
  if default_threshold is None:
    default_threshold = -np.inf

  thresholds = np.full(config.model_conf.n_species, default_threshold, np.float32)

  if pred_conf.custom_confidence_thresholds:
    for species_name, threshold in pred_conf.custom_confidence_thresholds.items():
      species_id = config.model_conf.species_list.index(species_name)
      thresholds[species_id] = threshold
  thresholds = thresholds[np.newaxis, :]
  thresholds.setflags(write=False)
  return thresholds


def create_species_blacklist(
  config: InferenceConfig, pred_config: PredictionConfig
) -> npt.NDArray:
  """Setup species filtering logic"""
  if pred_config.custom_species_list and len(pred_config.custom_species_list) > 0:
    species_ids_whitelist = np.empty(len(pred_config.custom_species_list), dtype=int)
    for i, species_name in enumerate(pred_config.custom_species_list):
      species_id = config.model_conf.species_list.index(species_name)
      species_ids_whitelist[i] = species_id

    species_whitelist = np.full(
      config.model_conf.n_species, fill_value=False, dtype=bool
    )
    species_whitelist[species_ids_whitelist] = True
  else:
    species_whitelist = np.full(
      config.model_conf.n_species, fill_value=True, dtype=bool
    )
  species_whitelist.setflags(write=False)

  species_blacklist = ~species_whitelist[np.newaxis, :]
  species_blacklist.setflags(write=False)
  return species_blacklist
