from __future__ import annotations

import os
from pathlib import Path

import psutil

from birdnet.acoustic_models.inference.encoding.benchmarking import (
  FullBenchmarkEmbMeta,
  MinimalBenchmarkEmbMeta,
)
from birdnet.acoustic_models.inference.encoding.result import (
  AcousticDataEncodingResult,
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet.acoustic_models.inference.encoding.tensor import AcousticEncodingTensor
from birdnet.acoustic_models.inference.encoding.worker import EmbeddingsWorker
from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.acoustic_models.inference_pipeline.configs import (
  EncodingConfig,
  InferenceConfig,
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


class EncodingStrategy(
  InferenceStrategyBase[
    AcousticEncodingResultBase, EncodingConfig, AcousticEncodingTensor
  ]
):
  def validate_config(
    self, config: InferenceConfig, specific_config: EncodingConfig
  ) -> None:
    pass

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
      EmbeddingsWorker(
        session_id=session_id,
        backend_loader=resources.worker_resources.backend_loader,
        device=resources.worker_resources.devices[i],
        batch_size=config.processing_conf.batch_size,
        wkr_ring_access_lock=resources.worker_resources.ring_access_lock,
        n_slots=config.processing_conf.n_slots,
        segment_duration_samples=config.model_conf.segment_size_samples,
        out_q=resources.worker_resources.results_queue,
        logging_queue=resources.logging_resources.logging_queue,
        logging_level=resources.logging_resources.logging_level,
        prd_all_done_event=resources.producer_resources.all_finished,
        rf_file_indices=resources.ring_buffer_resources.rf_file_indices,
        rf_segment_indices=resources.ring_buffer_resources.rf_segment_indices,
        rf_audio_samples=resources.ring_buffer_resources.rf_audio_samples,
        rf_batch_sizes=resources.ring_buffer_resources.rf_batch_sizes,
        rf_flags=resources.ring_buffer_resources.rf_flags,
        sem_fill=resources.ring_buffer_resources.sem_filled_slots,
        sem_free=resources.ring_buffer_resources.sem_free_slots,
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
    tensor: AcousticEncodingTensor,
    config: InferenceConfig,
    resources: PipelineResources,
    files: list[Path],
  ) -> AcousticEncodingResultBase:
    return AcousticFileEncodingResult(
      tensor=tensor,
      files=files,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      speed=config.processing_conf.speed,
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
    tensor: AcousticEncodingTensor,
    config: InferenceConfig,
    resources: PipelineResources,
  ) -> AcousticEncodingResultBase:
    return AcousticDataEncodingResult(
      tensor=tensor,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      speed=config.processing_conf.speed,
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
    specific_config: EncodingConfig,
    resources: PipelineResources,
    pred_result: AcousticEncodingResultBase,
  ) -> MinimalBenchmarkEmbMeta:
    assert resources.stats_resources.end_timepoint is not None
    assert resources.stats_resources.stop is not None
    wall_time_s = resources.stats_resources.stop - resources.stats_resources.start

    return MinimalBenchmarkEmbMeta(
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
      file_formats=get_file_formats({Path(x) for x in pred_result.inputs}),
    )

  def create_full_benchmark_meta(
    self,
    config: InferenceConfig,
    specific_config: EncodingConfig,
    resources: PipelineResources,
    pred_result: AcousticEncodingResultBase,
  ) -> FullBenchmarkEmbMeta:
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

    return FullBenchmarkEmbMeta(
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
      model_emb_dim=specific_config.emb_dim,
      _file_durations=resources.analyzer_resources.input_durations,
      file_segments_maximum=resources.analyzer_resources.max_segment_idx_ptr.value + 1,
      file_segments_total=resources.analyzer_resources.tot_n_segments_ptr.value,
      model_segment_duration_seconds=config.model_conf.segment_size_s,
      param_overlap_seconds=config.processing_conf.overlap_duration_s,
      param_batch_size=config.processing_conf.batch_size,
      param_prefetch_ratio=config.processing_conf.prefetch_ratio,
      mem_shm_ringsize=config.processing_conf.workers
      + (config.processing_conf.workers * config.processing_conf.prefetch_ratio),
      param_bandpass_fmin=config.filtering_conf.bandpass_fmin,
      param_bandpass_fmax=config.filtering_conf.bandpass_fmax,
      param_half_precision=config.processing_conf.half_precision,
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
      file_formats=get_file_formats({Path(x) for x in pred_result.inputs}),
      param_inference_library=config.model_conf.backend_kwargs.get(
        TF_BACKEND_LIB_ARG, NA
      ),
    )

  def get_benchmark_dir_name(self) -> str:
    return "encoding"

  def save_results_extra(
    self, result: AcousticEncodingResultBase, benchmark_run_out_dir: Path, prepend: str
  ) -> list[Path]:
    return []
