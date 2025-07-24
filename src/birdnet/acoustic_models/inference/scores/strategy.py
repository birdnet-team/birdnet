from __future__ import annotations

import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTrackingResult,
)
from birdnet.acoustic_models.inference.configs import (
  PredictionConfig,
  ScoresConfig,
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
from birdnet.acoustic_models.inference.scores.benchmarking import (
  FullBenchmarkMeta,
  MinimalBenchmarkMeta,
)
from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.inference.scores.tensor import ScoresTensor
from birdnet.acoustic_models.inference.scores.worker import ChildWorker
from birdnet.backends import (
  InferenceBackendLoader,
)
from birdnet.globals import (
  MODEL_TYPE_ACOUSTIC,
)


class ScoresStrategy(PredictionStrategy[PredictionResult, ScoresConfig, ScoresTensor]):
  def validate_config(
    self, config: PredictionConfig, specific_config: ScoresConfig
  ) -> None:
    if specific_config.apply_sigmoid:
      if specific_config.sigmoid_sensitivity is None:
        raise ValueError("sigmoid_sensitivity required when apply_sigmoid=True")
      if not 0.5 <= specific_config.sigmoid_sensitivity <= 1.5:
        raise ValueError("sigmoid_sensitivity must be in [0.5, 1.5]")

    if specific_config.custom_species_list:
      for species_name in specific_config.custom_species_list:
        if species_name not in config.model_conf.species_list:
          raise ValueError(f"Species '{species_name}' not in model's species list")

    if specific_config.custom_confidence_thresholds:
      for species_name in specific_config.custom_confidence_thresholds:
        if species_name not in config.model_conf.species_list:
          raise ValueError(f"Species '{species_name}' not in model's species list")

    if specific_config.top_k is not None and specific_config.top_k > len(
      config.model_conf.species_list
    ):
      raise ValueError(
        f"top_k cannot be larger than species count ({len(config.model_conf.species_list)})"
      )

  def create_tensor(
    self,
    config: PredictionConfig,
    specific_config: ScoresConfig,
    memory_layout: MemoryLayout,
  ) -> ScoresTensor:
    prob_dtype = np.float16 if config.processing_conf.half_precision else np.float32
    n_species = len(config.model_conf.species_list)
    top_k = specific_config.top_k if specific_config.top_k is not None else n_species

    return ScoresTensor(
      memory_layout.n_files,
      n_segments=memory_layout.reserve_n_segments,
      top_k=top_k,
      n_species=n_species,
      prob_dtype=prob_dtype,
      segment_indices_dtype=memory_layout.rf_segment_indices.dtype,
      files_dtype=memory_layout.rf_file_indices.dtype,
      max_segment_index=memory_layout.max_segment_idx_ptr,
    )

  def create_workers(
    self,
    config: PredictionConfig,
    specific_config: ScoresConfig,
    devices: list[str],
    backend_loader: InferenceBackendLoader,
    shared_resources: SharedResources,
  ) -> list[mp.Process]:
    n_species = len(config.model_conf.species_list)
    species_whitelist, thresholds = self._setup_species_filtering(
      config.model_conf.species_list, specific_config, n_species
    )

    species_blacklist = ~species_whitelist[np.newaxis, :]
    species_blacklist.setflags(write=False)
    species_thresholds = thresholds[np.newaxis, :]
    species_thresholds.setflags(write=False)

    top_k = specific_config.top_k if specific_config.top_k is not None else n_species

    return [
      mp.Process(
        target=ChildWorker(
          backend_loader=backend_loader,
          device=devices[i],
          top_k=top_k,
          species_thresholds=species_thresholds,
          species_blacklist=species_blacklist,
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
          apply_sigmoid=specific_config.apply_sigmoid,
          prob_dtype=shared_resources.result_dtype,
          sigmoid_sensitivity=specific_config.sigmoid_sensitivity,
          wkr_stats_queue=shared_resources.wkr_stats_queue,
          track_performance=shared_resources.track_performance,
          cancel_event=shared_resources.cancel_event,
          sem_active_workers=shared_resources.sem_active_workers,
        ),
        name=f"ChildWorker-{i}",
        daemon=True,
      )
      for i in range(config.processing_conf.workers)
    ]

  def _setup_species_filtering(
    self,
    model_species_list: OrderedSet[str],
    scores_config: ScoresConfig,
    n_species: int,
  ):
    """Setup species filtering logic"""
    # Species whitelist
    if scores_config.custom_species_list and len(scores_config.custom_species_list) > 0:
      species_ids_whitelist = np.empty(
        len(scores_config.custom_species_list), dtype=int
      )
      for i, species_name in enumerate(scores_config.custom_species_list):
        species_id = model_species_list.index(species_name)
        species_ids_whitelist[i] = species_id

      species_whitelist = np.full(n_species, fill_value=False, dtype=bool)
      species_whitelist[species_ids_whitelist] = True
    else:
      species_whitelist = np.full(n_species, fill_value=True, dtype=bool)
    species_whitelist.setflags(write=False)

    # Thresholds
    default_threshold = scores_config.default_confidence_threshold
    if default_threshold is None:
      default_threshold = -np.inf

    thresholds = np.full(n_species, default_threshold, np.float32)

    if scores_config.custom_confidence_thresholds:
      for species_name, threshold in scores_config.custom_confidence_thresholds.items():
        species_id = model_species_list.index(species_name)
        thresholds[species_id] = threshold
    thresholds.setflags(write=False)

    return species_whitelist, thresholds

  def create_result(
    self,
    tensor: ScoresTensor,
    config: PredictionConfig,
    file_paths: OrderedSet[Path],
    file_durations: np.ndarray,
  ) -> PredictionResult:
    return PredictionResult(
      tensor=tensor,
      files=file_paths,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      species_list=config.model_conf.species_list,
      file_durations=file_durations,
    )

  def create_minimal_benchmark_meta(
    self,
    config: PredictionConfig,
    specific_config: ScoresConfig,
    pred_result: PredictionResult,
    processing_state: ProcessingState,
    start_timepoint: datetime,
    end_timepoint: datetime,
    wall_time_s: float,
    file_durations: np.ndarray,
    memory_layout: MemoryLayout,
  ) -> MinimalBenchmarkMeta:
    return MinimalBenchmarkMeta(
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
    specific_config: ScoresConfig,
    pred_result: PredictionResult,
    processing_state: ProcessingState,
    start_time: float,
    start_timepoint: datetime,
    end_timepoint: datetime,
    wall_time_s: float,
    file_durations: np.ndarray,
    memory_layout: MemoryLayout,
    perf_result: PerformanceTrackingResult,
  ) -> FullBenchmarkMeta:
    device_str = (
      ", ".join(config.processing_conf.device)
      if isinstance(config.processing_conf.device, list)
      else config.processing_conf.device
    )

    return FullBenchmarkMeta(
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
      _file_durations=file_durations,
      file_segments_maximum=memory_layout.max_segment_idx_ptr.value + 1,
      file_segments_total=processing_state.tot_n_segments_ptr.value,
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
      param_bandpass_use=config.filtering_conf.use_bandpass,
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
    return "scores"

  def save_results(
    self, result: PredictionResult, npz_path: Path, csv_path: Path
  ) -> str:
    print("Saving result using internal format (.npz)...")
    result.save(npz_path)
    print("Saving result using CSV format (.csv)...")
    result.to_csv(csv_path, encoding="utf-8", silent=False)
    return f"  {npz_path.absolute()}\n  {csv_path.absolute()}\n"


def predict_species_from_recordings(
  conf: PredictionConfig,
  scores_conf: ScoresConfig,
) -> PredictionResult:
  strategy = ScoresStrategy()
  return predict_from_recordings_generic(conf, strategy, scores_conf)
