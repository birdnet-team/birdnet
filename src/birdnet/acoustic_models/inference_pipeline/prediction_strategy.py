from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import psutil

from birdnet.acoustic_models.inference.prediction.benchmarking import (
  FullBenchmarkMeta,
  MinimalBenchmarkMeta,
)
from birdnet.acoustic_models.inference.prediction.result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
  AcousticPredictionResultBase,
)
from birdnet.acoustic_models.inference.prediction.tensor import AcousticPredictionTensor
from birdnet.acoustic_models.inference.prediction.worker import PredictionWorker
from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.acoustic_models.inference_pipeline.configs import (
  InferenceConfig,
  PredictionConfig,
)
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
)
from birdnet.acoustic_models.inference_pipeline.strategy import (
  InferenceStrategyBase,
)
from birdnet.backends import TF_BACKEND_LIB_ARG
from birdnet.globals import (
  MODEL_TYPE_ACOUSTIC,
  NA,
)
from birdnet.helper import get_file_formats


class PredictionStrategy(
  InferenceStrategyBase[
    AcousticPredictionResultBase, PredictionConfig, AcousticPredictionTensor
  ]
):
  def validate_config(
    self, config: InferenceConfig, specific_config: PredictionConfig
  ) -> None:
    if specific_config.apply_sigmoid:
      if specific_config.sigmoid_sensitivity is None:
        raise ValueError("sigmoid_sensitivity required when apply_sigmoid=True")
      if not 0.5 <= specific_config.sigmoid_sensitivity <= 1.5:
        raise ValueError("sigmoid_sensitivity must be in [0.5, 1.5]")

    if specific_config.custom_species_list:
      for species_name in specific_config.custom_species_list:
        if species_name not in config.model_conf.species_list:
          raise ValueError(f"species '{species_name}' not in model's species list")

    if specific_config.custom_confidence_thresholds:
      for species_name in specific_config.custom_confidence_thresholds:
        if species_name not in config.model_conf.species_list:
          raise ValueError(f"species '{species_name}' not in model's species list")

    if specific_config.top_k is not None and specific_config.top_k > len(
      config.model_conf.species_list
    ):
      raise ValueError(
        f"top_k cannot be larger than species count "
        f"({len(config.model_conf.species_list)})"
      )

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
        session_id=session_id,
        backend_loader=resources.worker_resources.backend_loader,
        device=resources.worker_resources.devices[i],
        top_k=top_k,
        species_thresholds=species_thresholds,
        species_blacklist=species_blacklist,
        batch_size=config.processing_conf.batch_size,
        wkr_ring_access_lock=resources.worker_resources.ring_access_lock,
        n_slots=config.processing_conf.n_slots,
        segment_duration_samples=config.model_conf.segment_size_samples,
        out_q=resources.worker_resources.results_queue,
        logging_queue=resources.logging_resources.logging_queue,
        logging_level=resources.logging_resources.logging_level,
        all_producers_finished=resources.producer_resources.all_finished,
        rf_file_indices=resources.ring_buffer_resources.rf_file_indices,
        rf_segment_indices=resources.ring_buffer_resources.rf_segment_indices,
        rf_audio_samples=resources.ring_buffer_resources.rf_audio_samples,
        rf_batch_sizes=resources.ring_buffer_resources.rf_batch_sizes,
        rf_flags=resources.ring_buffer_resources.rf_flags,
        sem_fill=resources.ring_buffer_resources.sem_filled_slots,
        sem_free=resources.ring_buffer_resources.sem_free_slots,
        apply_sigmoid=specific_config.apply_sigmoid,
        sigmoid_sensitivity=specific_config.sigmoid_sensitivity,
        wkr_stats_queue=resources.stats_resources.wkr_stats_queue,
        cancel_event=resources.processing_resources.cancel_event,
        sem_active_workers=resources.stats_resources.sem_active_workers,
        end_event=resources.processing_resources.end_event,
        start_signal=resources.worker_resources.start_signals[i],
        half_precision=config.processing_conf.half_precision,
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
      tensor=tensor,
      files=files,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      speed=config.processing_conf.speed,
      species_list=config.model_conf.species_list,
      file_durations=resources.analyzer_resources.input_durations,
      model_path=config.model_conf.path,
      model_fmin=config.model_conf.sig_fmin,
      model_fmax=config.model_conf.sig_fmax,
      model_sr=config.model_conf.sample_rate,
      model_precision=config.model_conf.backend_type.precision(),
      model_version=config.model_conf.version,
    )

  def create_array_result(
    self,
    tensor: AcousticPredictionTensor,
    config: InferenceConfig,
    resources: PipelineResources,
  ) -> AcousticPredictionResultBase:
    return AcousticDataPredictionResult(
      tensor=tensor,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      speed=config.processing_conf.speed,
      species_list=config.model_conf.species_list,
      input_durations=resources.analyzer_resources.input_durations,
      model_path=config.model_conf.path,
      model_fmin=config.model_conf.sig_fmin,
      model_fmax=config.model_conf.sig_fmax,
      model_sr=config.model_conf.sample_rate,
      model_precision=config.model_conf.backend_type.precision(),
      model_version=config.model_conf.version,
    )

  def create_minimal_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: PredictionConfig,
    resources: PipelineResources,
    pred_result: AcousticPredictionResultBase,
  ) -> MinimalBenchmarkMeta:
    assert resources.stats_resources.end_timepoint is not None
    assert resources.stats_resources.stop is not None
    wall_time_s = resources.stats_resources.stop - resources.stats_resources.start

    file_formats = NA
    if isinstance(pred_result, AcousticFilePredictionResult):
      file_formats = get_file_formats({Path(x) for x in pred_result.inputs})

    return MinimalBenchmarkMeta(
      _start_timepoint=resources.stats_resources.start_timepoint,
      _end_timepoint=resources.stats_resources.end_timepoint,
      _time_wall_time_s=wall_time_s,
      _file_durations=resources.analyzer_resources.input_durations,
      mem_result_total_memory_usage_MiB=pred_result.memory_size_mb,
      mem_shm_size_file_indices_MiB=resources.ring_buffer_resources.rf_file_indices.nbytes
      / 1024**2,
      mem_shm_size_segment_indices_MiB=resources.ring_buffer_resources.rf_segment_indices.nbytes
      / 1024**2,
      mem_shm_size_audio_samples_MiB=resources.ring_buffer_resources.rf_audio_samples.nbytes
      / 1024**2,
      mem_shm_size_batch_sizes_MiB=resources.ring_buffer_resources.rf_batch_sizes.nbytes
      / 1024**2,
      mem_shm_size_flags_MiB=resources.ring_buffer_resources.rf_flags.nbytes / 1024**2,
      file_segments_total=resources.analyzer_resources.tot_n_segments_ptr.value,
      model_segment_duration_seconds=config.model_conf.segment_size_s,
      file_formats=file_formats,
    )

  def create_full_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: PredictionConfig,
    resources: PipelineResources,
    pred_result: AcousticPredictionResultBase,
  ) -> FullBenchmarkMeta:
    perf_result = resources.stats_resources.tracking_result
    assert perf_result is not None

    assert resources.stats_resources.end_timepoint is not None
    assert resources.stats_resources.stop is not None
    wall_time_s = resources.stats_resources.stop - resources.stats_resources.start

    device_str = (
      ", ".join(config.processing_conf.device)
      if isinstance(config.processing_conf.device, list)
      else config.processing_conf.device
    )

    file_formats = NA
    if isinstance(pred_result, AcousticFilePredictionResult):
      file_formats = get_file_formats({Path(x) for x in pred_result.inputs})

    return FullBenchmarkMeta(
      _start_timepoint=resources.stats_resources.start_timepoint,
      _end_timepoint=resources.stats_resources.end_timepoint,
      param_producers=config.processing_conf.feeders,
      param_workers=config.processing_conf.workers,
      _worker_avg_wall_time_s=perf_result.worker_avg_wall_time_s,
      param_devices=device_str,
      model_type=MODEL_TYPE_ACOUSTIC,
      model_version=config.model_conf.version,
      model_is_custom=config.model_conf.is_custom,
      model_path=str(config.model_conf.path.absolute()),
      model_species=len(config.model_conf.species_list),
      model_precision=config.model_conf.backend_type.precision(),
      _file_durations=resources.analyzer_resources.input_durations,
      file_segments_maximum=resources.analyzer_resources.max_segment_idx_ptr.value + 1,
      file_segments_total=resources.analyzer_resources.tot_n_segments_ptr.value,
      model_segment_duration_seconds=config.model_conf.segment_size_s,
      param_overlap_seconds=config.processing_conf.overlap_duration_s,
      param_batch_size=config.processing_conf.batch_size,
      param_top_k=specific_config.top_k,
      param_prefetch_ratio=config.processing_conf.prefetch_ratio,
      mem_shm_ringsize=config.processing_conf.workers
      + (config.processing_conf.workers * config.processing_conf.prefetch_ratio),
      param_sigmoid_apply=specific_config.apply_sigmoid,
      param_sigmoid_sensitivity=specific_config.sigmoid_sensitivity
      if specific_config.apply_sigmoid
      else None,
      param_bandpass_fmin=config.filtering_conf.bandpass_fmin,
      param_bandpass_fmax=config.filtering_conf.bandpass_fmax,
      param_half_precision=config.processing_conf.half_precision,
      param_confidence_threshold_default=specific_config.default_confidence_threshold,
      param_custom_species=len(specific_config.custom_species_list)
      if specific_config.custom_species_list
      else 0,
      param_confidence_threshold_custom=len(
        specific_config.custom_confidence_thresholds
      )
      if specific_config.custom_confidence_thresholds
      else 0,
      _time_rampup_first_line_s=resources.stats_resources.start_time
      - psutil.Process(os.getpid()).create_time(),  # TODO: Berechnen
      _time_wall_time_s=wall_time_s,
      mem_result_total_memory_usage_MiB=pred_result.memory_size_mb,
      mem_shm_size_file_indices_MiB=resources.ring_buffer_resources.rf_file_indices.nbytes
      / 1024**2,
      mem_shm_size_segment_indices_MiB=resources.ring_buffer_resources.rf_segment_indices.nbytes
      / 1024**2,
      mem_shm_size_audio_samples_MiB=resources.ring_buffer_resources.rf_audio_samples.nbytes
      / 1024**2,
      mem_shm_size_batch_sizes_MiB=resources.ring_buffer_resources.rf_batch_sizes.nbytes
      / 1024**2,
      mem_shm_size_flags_MiB=resources.ring_buffer_resources.rf_flags.nbytes / 1024**2,
      mem_memory_usage_maximum_MiB=perf_result.max_memory_usages_MiB,
      mem_memory_usage_average_MiB=perf_result.avg_memory_usages_MiB,
      cpu_usage_maximum_pct=perf_result.max_cpu_usages_pct,
      cpu_usage_average_pct=perf_result.avg_cpu_usages_pct,
      mem_shm_slots_average_free=perf_result.avg_free_slots,
      mem_shm_slots_average_busy=perf_result.avg_busy_slots,
      mem_shm_slots_average_buffered=perf_result.avg_preloaded_slots,
      worker_busy_average=perf_result.avg_busy_workers,
      file_batches_processed=perf_result.total_batches_processed,
      speed_worker_xrt=perf_result.worker_speed_xrt,
      speed_worker_xrt_max=perf_result.worker_speed_xrt_max,
      model_backend=config.model_conf.backend_type.name(),
      model_sample_rate=config.model_conf.sample_rate,
      model_sig_fmin=config.model_conf.sig_fmin,
      model_sig_fmax=config.model_conf.sig_fmax,
      worker_wait_time_average_milliseconds=perf_result.avg_wait_time_ms,
      file_formats=file_formats,
      param_inference_library=config.model_conf.backend_kwargs.get(
        TF_BACKEND_LIB_ARG, NA
      ),
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
  # Species whitelist
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
