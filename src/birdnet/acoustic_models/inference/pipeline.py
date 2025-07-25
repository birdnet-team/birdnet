from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import threading
import time
from dataclasses import asdict
from typing import cast

import numpy as np

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
  validate_common_config,
)
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.files_analyzer import FilesAnalyzer
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTracker,
  PerformanceTrackingResult,
)
from birdnet.acoustic_models.inference.producer import Producer
from birdnet.acoustic_models.inference.resources import (
  LoggingResources,
  PipelineResources,
  ResourceManager,
)
from birdnet.acoustic_models.inference.strategy import (
  PredictionStrategy,
)
from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.globals import WRITABLE_FLAG
from birdnet.helper import create_shm_ring


def predict_from_recordings_generic(
  conf: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
) -> ResultType:
  validate_common_config(conf)
  strategy.validate_config(conf, specific_config)

  resource_manager = ResourceManager(conf, strategy.get_benchmark_dir_name())
  resources = resource_manager.create_all_resources()

  logging_thread = _start_logging(resources)

  result_tensor = strategy.create_tensor(
    conf,
    specific_config,
    resources.ring_buffer_resources,
    resources.analyzer_resources,
  )

  try:
    with (
      create_shm_ring(resources.ring_buffer_resources.rf_file_indices),
      create_shm_ring(resources.ring_buffer_resources.rf_segment_indices),
      create_shm_ring(resources.ring_buffer_resources.rf_audio_samples),
      create_shm_ring(resources.ring_buffer_resources.rf_batch_sizes),
      create_shm_ring(resources.ring_buffer_resources.rf_flags) as shm_ring_flags,
    ):
      flags = resources.ring_buffer_resources.rf_flags.get_array(shm_ring_flags)
      flags[:] = WRITABLE_FLAG

      file_analyzer_thread = _start_file_analyzer(conf, resources)

      producer_processes = _start_producers(conf, resources)

      worker_processes = _start_workers(conf, strategy, specific_config, resources)

      perf_tracker_process = (
        _start_performance_tracker(conf, resources)
        if resources.stats_resources.track_performance
        else None
      )

      _run_consumer(conf, result_tensor, resources)

      file_durations, perf_result = _cleanup_processes(
        file_analyzer_thread,
        producer_processes,
        worker_processes,
        perf_tracker_process,
        resources,
      )

    resources.stats_resources.mark_stop()

    if resources.processing_state.cancel_event.is_set():
      raise RuntimeError(
        f"Analysis was cancelled due to an error. Please check the logs: {resources.logging_resources.log_file.absolute()}"
      )

    result = strategy.create_result(
      result_tensor, conf, resources.analyzer_resources.file_paths, file_durations
    )

    _handle_statistics(
      conf, strategy, specific_config, result, file_durations, perf_result, resources
    )

    return result

  finally:
    _cleanup_logging(resources.logging_resources, logging_thread)


def _start_logging(resources: PipelineResources) -> threading.Thread:
  logging_listener = threading.Thread(
    target=bn_logging.QueueFileWriter(
      log_queue=resources.logging_resources.logging_queue,
      logging_level=resources.logging_resources.logging_level,
      log_file=resources.logging_resources.log_file,
      cancel_event=resources.processing_state.cancel_event,
      stop_event=resources.logging_resources.stop_logging_event,
      processing_finished_event=resources.processing_state.processing_finished_event,
    ),
    name="QueueFileWriter",
    daemon=True,
  )
  logging_listener.start()
  return logging_listener


def _start_performance_tracker(
  config: PredictionConfig, resources: PipelineResources
) -> mp.Process:
  assert resources.stats_resources.track_performance
  assert resources.stats_resources.sem_active_workers is not None
  assert resources.stats_resources.perf_res_queue is not None
  assert resources.stats_resources.wkr_stats_queue is not None
  assert resources.stats_resources.prd_stats_queue is not None

  perf_tracker = mp.Process(
    target=PerformanceTracker(
      pred_dur_queue=resources.stats_resources.wkr_stats_queue,
      processing_finished_event=resources.processing_state.processing_finished_event,
      update_interval=0.5,
      print_interval=1,
      prod_stats_queue=resources.stats_resources.prd_stats_queue,
      n_workers=config.processing_conf.workers,
      start=resources.stats_resources.start,
      sem_filled_slots=resources.ring_buffer_resources.sem_filled_slots,
      workers_start=time.perf_counter(),
      segment_size_s=config.model_conf.segment_size_s,
      logging_queue=resources.logging_resources.logging_queue,
      logging_level=resources.logging_resources.logging_level,
      perf_res=resources.stats_resources.perf_res_queue,
      parent_process_id=os.getpid(),
      rf_flags=resources.ring_buffer_resources.rf_flags,
      tot_n_segments_ptr=resources.analyzer_resources.tot_n_segments_ptr,
      cancel_event=resources.processing_state.cancel_event,
      sem_active_workers=resources.stats_resources.sem_active_workers,
    ),
    name="PerformanceTracker",
    daemon=True,
  )
  perf_tracker.start()

  return perf_tracker


def _start_file_analyzer(
  config: PredictionConfig,
  resources: PipelineResources,
) -> threading.Thread:
  file_analyzer_proc = threading.Thread(
    target=FilesAnalyzer(
      files=resources.analyzer_resources.file_paths,
      logging_level=resources.logging_resources.logging_level,
      logging_queue=resources.logging_resources.logging_queue,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      max_segment_idx_ptr=resources.analyzer_resources.max_segment_idx_ptr,
      rf_segment_indices=resources.ring_buffer_resources.rf_segment_indices,
      analyzing_result=resources.analyzer_resources.analyzer_queue,
      tot_n_segments=resources.analyzer_resources.tot_n_segments_ptr,
      cancel_event=resources.processing_state.cancel_event,
    ),
    name="FileAnalyzer",
    daemon=True,
  )
  file_analyzer_proc.start()
  return file_analyzer_proc


def _start_producers(
  config: PredictionConfig,
  resources: PipelineResources,
) -> list[mp.Process]:
  producer_processes = [
    mp.Process(
      target=Producer(
        files_queue=resources.producer_resources.files_queue,
        batch_size=config.processing_conf.batch_size,
        prd_all_done_event=resources.producer_resources.prd_all_done_event,
        n_slots=config.processing_conf.n_slots,
        prd_ring_access_lock=resources.producer_resources.ring_access_lock,
        prod_stats_queue=resources.stats_resources.prd_stats_queue,
        rf_file_indices=resources.ring_buffer_resources.rf_file_indices,
        rf_segment_indices=resources.ring_buffer_resources.rf_segment_indices,
        rf_audio_samples=resources.ring_buffer_resources.rf_audio_samples,
        rf_batch_sizes=resources.ring_buffer_resources.rf_batch_sizes,
        rf_flags=resources.ring_buffer_resources.rf_flags,
        logging_queue=resources.logging_resources.logging_queue,
        logging_level=resources.logging_resources.logging_level,
        sem_free_slots=resources.ring_buffer_resources.sem_free_slots,
        sem_filled_slots=resources.ring_buffer_resources.sem_filled_slots,
        segment_duration_s=config.model_conf.segment_size_s,
        overlap_duration_s=config.processing_conf.overlap_duration_s,
        target_sample_rate=config.model_conf.sample_rate,
        use_bandpass=config.filtering_conf.use_bandpass,
        bandpass_fmax=config.filtering_conf.bandpass_fmax,
        bandpass_fmin=config.filtering_conf.bandpass_fmin,
        fmin=config.model_conf.sig_fmin,
        fmax=config.model_conf.sig_fmax,
        max_segment_idx_ptr=resources.analyzer_resources.max_segment_idx_ptr,
        prod_done_ptr=resources.producer_resources.n_finished_pointer,
        n_feeders=resources.producer_resources.n_producers,
        cancel_event=resources.processing_state.cancel_event,
      ),
      name=f"ChildProducer-{i}",
      daemon=True,
    )
    for i in range(resources.producer_resources.n_producers)
  ]

  for p in producer_processes:
    p.start()

  return producer_processes


def _start_workers(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
  resources: PipelineResources,
) -> list[mp.Process]:
  try:
    resources.worker_resources.backend_loader.on_before_worker_initialized()
  except Exception as exc:
    raise RuntimeError(f"Error during backend initialization: {exc}")

  worker_processes = strategy.create_workers(
    config,
    specific_config,
    resources,
  )
  for w in worker_processes:
    w.start()
  return worker_processes


def _run_consumer(
  config: PredictionConfig, result_tensor: TensorBase, resources: PipelineResources
) -> None:
  consumer = Consumer(
    n_workers=config.processing_conf.workers,
    worker_queue=resources.worker_resources.results_queue,
    tensor=result_tensor,
    cancel_event=resources.processing_state.cancel_event,
  )
  consumer()


def _cleanup_processes(
  file_analyzer_proc: threading.Thread,
  producer_processes: list[mp.Process],
  worker_processes: list[mp.Process],
  perf_tracking_process: mp.Process | None,
  resources: PipelineResources,
) -> tuple[np.ndarray, PerformanceTrackingResult | None]:
  logger = bn_logging.get_logger(__name__)

  resources.processing_state.processing_finished_event.set()

  file_durations = np.array(
    cast(list[float], resources.analyzer_resources.analyzer_queue.get()),
    dtype=np.float16,
  )
  resources.analyzer_resources.analyzer_queue.close()
  file_analyzer_proc.join()
  logger.debug("File analyzer finished.")

  for p in producer_processes:
    p.join()
    logger.debug(f"Producer '{p.name}' finished.")
  logger.debug("All producers finished.")

  for w in worker_processes:
    w.join()
    logger.debug(f"Worker '{w.name}' finished.")
  logger.debug("All workers finished.")

  perf_result = None
  if resources.stats_resources.track_performance:
    assert resources.stats_resources.perf_res_queue is not None
    assert perf_tracking_process is not None
    perf_result = cast(
      PerformanceTrackingResult, resources.stats_resources.perf_res_queue.get()
    )
    perf_tracking_process.join()
    logger.debug("Performance tracker finished.")

  return file_durations, perf_result


def _handle_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
  result: ResultType,
  file_durations: np.ndarray,
  perf_result: PerformanceTrackingResult | None,
  resources: PipelineResources,
) -> None:
  if config.output_conf.show_stats in ("minimal", "progress"):
    _show_minimal_statistics(
      config,
      strategy,
      resources,
      specific_config,
      result,
      file_durations,
    )
  elif config.output_conf.show_stats == "benchmark":
    assert perf_result is not None
    _create_benchmark_statistics(
      config,
      strategy,
      resources,
      specific_config,
      result,
      file_durations,
      perf_result,
    )


def _show_minimal_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  resources: PipelineResources,
  specific_config: ConfigType,
  result: ResultType,
  file_durations: np.ndarray,
) -> None:
  bmm = strategy.create_minimal_benchmark_meta(
    config, specific_config, resources, result, file_durations
  )

  summary = (
    f"-------------------------------\n"
    f"----------- Summary -----------\n"
    f"-------------------------------\n"
    f"Start time: {bmm.time_begin}\n"
    f"End time:   {bmm.time_end}\n"
    f"Wall time:  {bmm.time_wall_time}\n"
    f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
    f"  Total duration: {bmm.file_duration_sum}\n"
    f"  Average duration: {bmm.file_duration_average}\n"
    f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
    f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
    f"Memory usage:\n"
    f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
    f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
    f"Performance:\n"
    f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
    f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
  )
  print(summary)


def _create_benchmark_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  resources: PipelineResources,
  specific_config: ConfigType,
  result: ResultType,
  file_durations: np.ndarray,
  perf_result: PerformanceTrackingResult,
) -> None:
  bmm = strategy.create_full_benchmark_meta(
    config, specific_config, resources, result, file_durations, perf_result
  )

  benchmark_dir = resources.stats_resources.benchmark_dir
  benchmark_run_out_dir = resources.stats_resources.benchmark_run_dir
  iso_time = resources.stats_resources.start_iso_time

  assert benchmark_dir is not None
  assert benchmark_run_out_dir is not None

  meta_df_out = benchmark_dir / "runs.csv"
  stats_out_json = benchmark_run_out_dir / f"stats-{iso_time}.json"
  stats_human_readable_out = benchmark_run_out_dir / f"stats-{iso_time}.txt"
  result_npz = benchmark_run_out_dir / f"result-{iso_time}.npz"

  bm = asdict(bmm)
  del_keys = [k for k in bm if k.startswith("_")]
  for k in del_keys:
    del bm[k]
  bm = bmm.to_dict()

  with open(stats_out_json, "w", encoding="utf8") as f:
    json.dump(bm, f, indent=2, ensure_ascii=False)

  import pandas as pd

  meta_df = pd.DataFrame.from_records([bm])
  meta_df.to_csv(meta_df_out, mode="a", header=not meta_df_out.exists(), index=False)

  summary = (
    f"-------------------------------\n"
    f"------ Benchmark summary ------\n"
    f"-------------------------------\n"
    f"Start time: {bmm.time_begin}\n"
    f"End time:   {bmm.time_end}\n"
    f"Wall time:  {bmm.time_wall_time}\n"
    f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
    f"  Total duration: {bmm.file_duration_sum}\n"
    f"  Average duration: {bmm.file_duration_average}\n"
    f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
    f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
    f"Feeder(s): {bmm.param_producers}\n"
    f"Buffer: {bmm.mem_shm_slots_average_filled:.1f}/{config.processing_conf.n_slots} filled slots (mean)\n"
    f"Busy workers: {bmm.worker_busy_average:.1f}/{bmm.param_workers} (mean)\n"
    f"  Average wait time for next batch: {bmm.worker_wait_time_average_milliseconds:.3f} ms\n"
    f"Memory usage:\n"
    f"  Program: {bmm.mem_memory_usage_maximum_MiB:.2f} M (total max)\n"
    f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
    f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
    f"Performance:\n"
    f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
    f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
    f"Worker performance:\n"
    f"  {bmm.speed_worker_xrt:.0f} x real-time (RTF: {bmm.speed_worker_rtf:.8f})\n"
    f"  {bmm.speed_worker_total_seg_per_second:.0f} segments/s ({bmm.speed_worker_total_audio_per_second} audio/s)\n"
  )

  stats_human_readable_out.write_text(summary, encoding="utf8")

  print("Saving result using internal format (.npz)...")
  result.save(result_npz)
  saved_files = [result_npz]
  saved_files += strategy.save_results_extra(result, benchmark_run_out_dir, iso_time)

  summary += (
    f"-------------------------------\n"
    f"Benchmark folder:\n"
    f"  {benchmark_run_out_dir.absolute()}\n"
    f"Statistics results written to:\n"
    f"  {stats_human_readable_out.absolute()}\n"
    f"  {stats_out_json.absolute()}\n"
    f"  {meta_df_out.absolute()}\n"
    f"Prediction results written to:\n"
  )
  for saved_file in saved_files:
    summary += f"  {saved_file.absolute()}\n"
  summary += (
    f"Log file written to:\n  {resources.logging_resources.log_file.absolute()}\n"
  )

  print(summary)


def _cleanup_logging(
  logging_resources: LoggingResources,
  logging_thread: threading.Thread,
) -> None:
  logging_resources.stop_logging_event.set()
  logging_thread.join()
  logging_resources.logging_queue.close()
  logging_resources.logging_queue.join_thread()
  bn_logging.remove_queue_handler(logging_resources.queue_handler)

  shutil.copyfile(logging_resources.log_file, logging_resources.global_log_file)
