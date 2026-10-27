from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic

import psutil

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
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.acoustic.inference.core.worker import WorkerBase
from birdnet.acoustic.inference.resources import (
  PipelineResources,
)
from birdnet.core.backends import TF_BACKEND_LIB_ARG
from birdnet.core.base import ResultBase
from birdnet.globals import (
  MODEL_TYPE_ACOUSTIC,
  NA,
)
from birdnet.utils.audio_files import get_file_formats


class InferenceStrategyBase(Generic[ResultType, ConfigType, TensorType], ABC):
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

  def _result_kwargs(self, config: InferenceConfig) -> dict[str, Any]:
    return {
      "segment_duration_s": config.model_conf.segment_size_s,
      "overlap_duration_s": config.processing_conf.overlap_duration_s,
      "speed": config.processing_conf.speed,
      "model_path": config.model_conf.path,
      "model_fmin": config.model_conf.sig_fmin,
      "model_fmax": config.model_conf.sig_fmax,
      "model_sr": config.model_conf.sample_rate,
      "model_precision": config.model_conf.backend_type.precision(),
      "model_version": config.model_conf.version,
    }

  def _worker_kwargs(
    self,
    session_id: str,
    config: InferenceConfig,
    resources: PipelineResources,
    worker_index: int,
  ) -> dict[str, Any]:
    return {
      "session_id": session_id,
      "backend_loader": resources.worker_resources.backend_loader,
      "device": resources.worker_resources.devices[worker_index],
      "batch_size": config.processing_conf.batch_size,
      "wkr_ring_access_lock": resources.worker_resources.ring_access_lock,
      "n_slots": config.processing_conf.n_slots,
      "segment_duration_samples": config.model_conf.segment_size_samples,
      "out_q": resources.worker_resources.results_queue,
      "logging_queue": resources.logging_resources.logging_queue,
      "logging_level": resources.logging_resources.logging_level,
      "all_producers_finished": resources.producer_resources.all_finished,
      "rf_file_indices": resources.ring_buffer_resources.rf_file_indices,
      "rf_segment_indices": resources.ring_buffer_resources.rf_segment_indices,
      "rf_audio_samples": resources.ring_buffer_resources.rf_audio_samples,
      "rf_batch_sizes": resources.ring_buffer_resources.rf_batch_sizes,
      "rf_flags": resources.ring_buffer_resources.rf_flags,
      "sem_fill": resources.ring_buffer_resources.sem_filled_slots,
      "sem_free": resources.ring_buffer_resources.sem_free_slots,
      "wkr_stats_queue": resources.stats_resources.wkr_stats_queue,
      "cancel_event": resources.processing_resources.cancel_event,
      "sem_active_workers": resources.stats_resources.sem_active_workers,
      "end_event": resources.processing_resources.end_event,
      "start_signal": resources.worker_resources.start_signals[worker_index],
      "finish_signal": resources.worker_resources.finish_signals[worker_index],
      "half_precision": config.processing_conf.half_precision,
      "start_method": config.start_method,
    }

  def _file_formats(self, result: ResultBase) -> str:
    # An array result's `inputs` are array indices, not paths.
    if isinstance(result, (AcousticFilePredictionResult, AcousticFileEncodingResult)):
      return get_file_formats({Path(x) for x in result.inputs})
    return NA

  def _minimal_meta_kwargs(
    self,
    config: InferenceConfig,
    resources: PipelineResources,
    result: ResultType,
  ) -> dict[str, Any]:
    assert resources.stats_resources.end_timepoint is not None
    assert resources.stats_resources.stop is not None
    ring = resources.ring_buffer_resources
    return {
      "_start_timepoint": resources.stats_resources.start_timepoint,
      "_end_timepoint": resources.stats_resources.end_timepoint,
      "_time_wall_time_s": resources.stats_resources.stop
      - resources.stats_resources.start,
      "_file_durations": resources.analyzer_resources.input_durations,
      "mem_result_total_memory_usage_MiB": result.memory_size_MiB,
      "mem_shm_size_file_indices_MiB": ring.rf_file_indices.nbytes / 1024**2,
      "mem_shm_size_segment_indices_MiB": ring.rf_segment_indices.nbytes / 1024**2,
      "mem_shm_size_audio_samples_MiB": ring.rf_audio_samples.nbytes / 1024**2,
      "mem_shm_size_batch_sizes_MiB": ring.rf_batch_sizes.nbytes / 1024**2,
      "mem_shm_size_flags_MiB": ring.rf_flags.nbytes / 1024**2,
      "file_segments_total": resources.analyzer_resources.tot_n_segments_ptr.value,
      "model_segment_duration_seconds": config.model_conf.segment_size_s,
      "file_formats": self._file_formats(result),
    }

  def _full_meta_kwargs(
    self,
    config: InferenceConfig,
    resources: PipelineResources,
    result: ResultType,
  ) -> dict[str, Any]:
    perf_result = resources.stats_resources.tracking_result
    assert perf_result is not None

    device_str = (
      ", ".join(config.processing_conf.device)
      if isinstance(config.processing_conf.device, list)
      else config.processing_conf.device
    )

    return {
      **self._minimal_meta_kwargs(config, resources, result),
      "sw_start_method": config.start_method,
      "param_producers": config.processing_conf.producers,
      "param_workers": config.processing_conf.workers,
      "_worker_avg_wall_time_s": perf_result.worker_avg_wall_time_s,
      "param_devices": device_str,
      "model_type": MODEL_TYPE_ACOUSTIC,
      "model_version": config.model_conf.version,
      "model_is_custom": config.model_conf.is_custom,
      "model_path": str(config.model_conf.path.absolute()),
      "model_species": len(config.model_conf.species_list),
      "model_precision": config.model_conf.backend_type.precision(),
      "file_segments_maximum": resources.analyzer_resources.max_segment_idx_ptr.value
      + 1,
      "param_overlap_seconds": config.processing_conf.overlap_duration_s,
      "param_batch_size": config.processing_conf.batch_size,
      "param_prefetch_ratio": config.processing_conf.prefetch_ratio,
      "mem_shm_ringsize": config.processing_conf.n_slots,
      "param_bandpass_fmin": config.filtering_conf.bandpass_fmin,
      "param_bandpass_fmax": config.filtering_conf.bandpass_fmax,
      "param_half_precision": config.processing_conf.half_precision,
      "_time_rampup_first_line_s": resources.stats_resources.start_time
      - psutil.Process(os.getpid()).create_time(),  # TODO: measure the real ramp-up
      "mem_memory_usage_maximum_MiB": perf_result.max_memory_usages_MiB,
      "mem_memory_usage_average_MiB": perf_result.avg_memory_usages_MiB,
      "cpu_usage_maximum_pct": perf_result.max_cpu_usages_pct,
      "cpu_usage_average_pct": perf_result.avg_cpu_usages_pct,
      "mem_shm_slots_average_free": perf_result.avg_free_slots,
      "mem_shm_slots_average_busy": perf_result.avg_busy_slots,
      "mem_shm_slots_average_buffered": perf_result.avg_preloaded_slots,
      "worker_busy_average": perf_result.avg_busy_workers,
      "file_batches_processed": perf_result.total_batches_processed,
      "speed_worker_xrt": perf_result.worker_speed_xrt,
      "speed_worker_xrt_max": perf_result.worker_speed_xrt_max,
      "model_backend": config.model_conf.backend_type.name(),
      "model_sample_rate": config.model_conf.sample_rate,
      "model_sig_fmin": config.model_conf.sig_fmin,
      "model_sig_fmax": config.model_conf.sig_fmax,
      "worker_wait_time_average_milliseconds": perf_result.avg_wait_time_ms,
      "param_inference_library": config.model_conf.backend_kwargs.get(
        TF_BACKEND_LIB_ARG, NA
      ),
    }
