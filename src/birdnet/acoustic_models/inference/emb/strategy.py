from __future__ import annotations

import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.emb.benchmarking import (
  FullBenchmarkEmbMeta,
  MinimalBenchmarkEmbMeta,
)
from birdnet.acoustic_models.inference.emb.prediction_result import (
  EmbeddingsPredictionResult,
)
from birdnet.acoustic_models.inference.emb.tensor import EmbeddingsTensor
from birdnet.acoustic_models.inference.emb.worker import EmbeddingsWorker
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTrackingResult,
)
from birdnet.acoustic_models.inference.configs import (
  EmbeddingsConfig,
  PredictionConfig,
)
from birdnet.acoustic_models.inference.pipeline import (
  predict_from_recordings_generic,
)
from birdnet.acoustic_models.inference.states import (
  MemoryLayout,
  ProcessingState,
  SharedResources,
)
from birdnet.acoustic_models.inference.strategy import (
  PredictionStrategy,
  get_file_formats,
)
from birdnet.backends import (
  InferenceBackendLoader,
)
from birdnet.globals import (
  MODEL_TYPE_ACOUSTIC,
)


class EmbeddingsStrategy(
  PredictionStrategy[EmbeddingsPredictionResult, EmbeddingsConfig, EmbeddingsTensor]
):
  def validate_config(
    self, config: PredictionConfig, specific_config: EmbeddingsConfig
  ) -> None:
    pass

  def create_tensor(
    self,
    config: PredictionConfig,
    specific_config: EmbeddingsConfig,
    memory_layout: MemoryLayout,
  ) -> EmbeddingsTensor:
    emb_dtype = np.float16 if config.processing_conf.half_precision else np.float32

    return EmbeddingsTensor(
      memory_layout.n_files,
      n_segments=memory_layout.reserve_n_segments,
      emb_dim=specific_config.emb_dim,
      emb_dtype=emb_dtype,
      segment_indices_dtype=memory_layout.rf_segment_indices.dtype,
      files_dtype=memory_layout.rf_file_indices.dtype,
      max_segment_index=memory_layout.max_segment_idx_ptr,
    )

  def create_workers(
    self,
    config: PredictionConfig,
    specific_config: EmbeddingsConfig,
    devices: list[str],
    backend_loader: InferenceBackendLoader,
    shared_resources: SharedResources,
  ) -> list[mp.Process]:
    return [
      mp.Process(
        target=EmbeddingsWorker(
          backend_loader=backend_loader,
          device=devices[i],
          batch_size=config.processing_conf.batch_size,
          wkr_ring_access_lock=shared_resources.wkr_ring_access_lock,
          n_slots=shared_resources.n_slots,
          segment_duration_samples=shared_resources.model_segment_size_samples,
          out_q=shared_resources.worker_queue,
          logging_queue=shared_resources.logging_queue,
          prd_all_done_event=shared_resources.prd_all_done_event,
          logging_level=shared_resources.logging_level,
          rf_file_indices=shared_resources.rf_file_indices,
          rf_segment_indices=shared_resources.rf_segment_indices,
          rf_audio_samples=shared_resources.rf_audio_samples,
          rf_batch_sizes=shared_resources.rf_batch_sizes,
          rf_flags=shared_resources.rf_flags,
          sem_fill=shared_resources.sem_filled_slots,
          sem_free=shared_resources.sem_free_slots,
          emb_dtype=shared_resources.result_dtype,
          wkr_stats_queue=shared_resources.wkr_stats_queue,
          track_performance=shared_resources.track_performance,
          cancel_event=shared_resources.cancel_event,
          sem_active_workers=shared_resources.sem_active_workers,
        ),
        name=f"EmbeddingsWorker-{i}",
        daemon=True,
      )
      for i in range(config.processing_conf.workers)
    ]

  def create_result(
    self,
    tensor: EmbeddingsTensor,
    config: PredictionConfig,
    file_paths: OrderedSet[Path],
    file_durations: np.ndarray,
  ) -> EmbeddingsPredictionResult:
    return EmbeddingsPredictionResult(
      tensor=tensor,
      files=file_paths,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      file_durations=file_durations,
    )

  def create_minimal_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: EmbeddingsConfig,
    pred_result: EmbeddingsPredictionResult,
    processing_state: ProcessingState,
    start_timepoint: datetime,
    end_timepoint: datetime,
    wall_time_s: float,
    file_durations: np.ndarray,
    memory_layout: MemoryLayout,
  ) -> MinimalBenchmarkEmbMeta:
    return MinimalBenchmarkEmbMeta(
      _start_timepoint=start_timepoint,
      _end_timepoint=end_timepoint,
      _time_wall_time_s=wall_time_s,
      _file_durations=file_durations,
      mem_result_total_memory_usage_MiB=pred_result.memory_size_mb,
      mem_shm_size_file_indices_MiB=memory_layout.rf_file_indices.nbytes / 1024**2,
      mem_shm_size_segment_indices_MiB=memory_layout.rf_segment_indices.nbytes
      / 1024**2,
      mem_shm_size_audio_samples_MiB=memory_layout.rf_audio_samples.nbytes / 1024**2,
      mem_shm_size_batch_sizes_MiB=memory_layout.rf_batch_sizes.nbytes / 1024**2,
      mem_shm_size_flags_MiB=memory_layout.rf_flags.nbytes / 1024**2,
      file_segments_total=processing_state.tot_n_segments_ptr.value,
      model_segment_duration_seconds=config.model_conf.segment_size_s,
      file_formats=get_file_formats(OrderedSet(Path(x) for x in pred_result.files)),
    )

  def create_full_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: EmbeddingsConfig,
    pred_result: EmbeddingsPredictionResult,
    processing_state: ProcessingState,
    start_time: float,
    start_timepoint: datetime,
    end_timepoint: datetime,
    wall_time_s: float,
    file_durations: np.ndarray,
    memory_layout: MemoryLayout,
    perf_result: PerformanceTrackingResult,
  ) -> FullBenchmarkEmbMeta:
    device_str = (
      ", ".join(config.processing_conf.device)
      if isinstance(config.processing_conf.device, list)
      else config.processing_conf.device
    )

    return FullBenchmarkEmbMeta(
      _start_timepoint=start_timepoint,
      _end_timepoint=end_timepoint,
      param_producers=config.processing_conf.feeders,
      param_workers=config.processing_conf.workers,
      _worker_avg_wall_time_s=perf_result.worker_avg_wall_time_s,
      param_devices=device_str,
      model_type=MODEL_TYPE_ACOUSTIC,
      model_version=config.model_conf.version,
      model_is_custom=config.model_conf.is_custom,
      model_path=str(config.model_conf.path.absolute()),
      model_species=len(config.model_conf.species_list),
      model_precision=config.model_conf.precision,
      model_emb_dim=specific_config.emb_dim,
      _file_durations=file_durations,
      file_segments_maximum=memory_layout.max_segment_idx_ptr.value + 1,
      file_segments_total=processing_state.tot_n_segments_ptr.value,
      model_segment_duration_seconds=config.model_conf.segment_size_s,
      param_overlap_seconds=config.processing_conf.overlap_duration_s,
      param_batch_size=config.processing_conf.batch_size,
      param_prefetch_ratio=config.processing_conf.prefetch_ratio,
      mem_shm_ringsize=config.processing_conf.workers
      + (config.processing_conf.workers * config.processing_conf.prefetch_ratio),
      param_bandpass_use=config.filtering_conf.use_bandpass,
      param_bandpass_fmin=config.filtering_conf.bandpass_fmin,
      param_bandpass_fmax=config.filtering_conf.bandpass_fmax,
      param_half_precision=config.processing_conf.half_precision,
      _time_rampup_first_line_s=start_time
      - psutil.Process(os.getpid()).create_time(),  # TODO: Berechnen
      _time_wall_time_s=wall_time_s,
      mem_result_total_memory_usage_MiB=pred_result.memory_size_mb,
      mem_shm_size_file_indices_MiB=memory_layout.rf_file_indices.nbytes / 1024**2,
      mem_shm_size_segment_indices_MiB=memory_layout.rf_segment_indices.nbytes
      / 1024**2,
      mem_shm_size_audio_samples_MiB=memory_layout.rf_audio_samples.nbytes / 1024**2,
      mem_shm_size_batch_sizes_MiB=memory_layout.rf_batch_sizes.nbytes / 1024**2,
      mem_shm_size_flags_MiB=memory_layout.rf_flags.nbytes / 1024**2,
      mem_memory_usage_maximum_MiB=perf_result.max_memory_usages_MiB,
      mem_memory_usage_average_MiB=perf_result.avg_memory_usages_MiB,
      cpu_usage_maximum_pct=perf_result.max_cpu_usages_pct,
      cpu_usage_average_pct=perf_result.avg_cpu_usages_pct,
      mem_shm_slots_average_free=perf_result.avg_free_slots,
      mem_shm_slots_average_busy=perf_result.avg_busy_slots,
      mem_shm_slots_average_buffered=perf_result.avg_preloaded_slots,
      worker_busy_average=perf_result.avg_busy_workers,
      _time_rampup_first_prediction_s=perf_result.ramp_up_time_until_first_pred_s,
      file_batches_processed=perf_result.total_batches_processed,
      speed_worker_xrt=perf_result.worker_speed_xrt,
      speed_worker_xrt_max=perf_result.worker_speed_xrt_max,
      model_backend=config.model_conf.backend,
      model_sample_rate=config.model_conf.sample_rate,
      model_sig_fmin=config.model_conf.sig_fmin,
      model_sig_fmax=config.model_conf.sig_fmax,
      worker_wait_time_average_milliseconds=perf_result.avg_wait_time_ms,
      file_formats=get_file_formats(OrderedSet(Path(x) for x in pred_result.files)),
      param_inference_library=config.model_conf.backend_kwargs.get("inference_library"),
    )

  def get_benchmark_dir_name(self) -> str:
    return "emb"

  def save_results(
    self, result: EmbeddingsPredictionResult, npz_path: Path, csv_path: Path
  ) -> str:
    print("Saving result using internal format (.npz)...")
    result.save(npz_path)
    return f"  {npz_path.absolute()}\n"


def predict_embeddings_from_recordings(
  conf: PredictionConfig,
  emb_config: EmbeddingsConfig,
) -> EmbeddingsPredictionResult:
  strategy = EmbeddingsStrategy()
  return predict_from_recordings_generic(conf, strategy, emb_config)
