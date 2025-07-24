from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.files_analyzer import FilesAnalyzer
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTrackingResult,
)
from birdnet.acoustic_models.inference.pipelining.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
  validate_common_config,
)
from birdnet.acoustic_models.inference.pipelining.states import (
  LoggingResources,
  MemoryLayout,
  PerformanceTrackingResources,
  ProcessingState,
  SharedResources,
  setup_logging,
  setup_memory_layout,
  setup_processing_state,
  setup_shared_resources,
  start_performance_tracker,
)
from birdnet.acoustic_models.inference.pipelining.strategy import (
  PredictionStrategy,
)
from birdnet.acoustic_models.inference.producer import ChildProducer
from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.backends import (
  InferenceBackendLoader,
  PBInferenceBackend,
  TFInferenceBackend,
)
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  PKG_NAME,
  WRITABLE_FLAG,
)
from birdnet.helper import (
  SF_FORMATS,
  create_shm_ring,
  get_supported_audio_files,
  uint_ctype_from_dtype,
  uint_dtype_for,
)


def predict_from_recordings_generic(
  conf: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
) -> ResultType:
  start = time.perf_counter()
  start_time = time.time()
  start_timepoint = datetime.now()

  validate_common_config(conf)
  strategy.validate_config(conf, specific_config)

  processing_state = setup_processing_state()
  logging_resources = setup_logging(
    conf,
    start_timepoint,
    processing_state,
    strategy.get_benchmark_dir_name(),
  )

  file_paths = _parse_input_files(conf.input_files)
  memory_layout = setup_memory_layout(conf, len(file_paths))
  shared_resources = setup_shared_resources(
    conf, memory_layout, logging_resources, processing_state, file_paths
  )

  result_tensor = strategy.create_tensor(conf, specific_config, memory_layout)

  try:
    with (
      create_shm_ring(memory_layout.rf_file_indices),
      create_shm_ring(memory_layout.rf_segment_indices),
      create_shm_ring(memory_layout.rf_audio_samples),
      create_shm_ring(memory_layout.rf_batch_sizes),
      create_shm_ring(memory_layout.rf_flags) as shm_ring_flags,
    ):
      flags = memory_layout.rf_flags.get_array(shm_ring_flags)
      flags[:] = WRITABLE_FLAG

      file_analyzer_proc = _start_file_analyzer(
        conf, file_paths, memory_layout, shared_resources, processing_state
      )

      producer_processes = _start_producers(conf, memory_layout, shared_resources)

      worker_processes = _start_workers(
        conf, strategy, specific_config, shared_resources
      )

      perf_tracking_resources = (
        start_performance_tracker(conf, shared_resources, processing_state, start)
        if shared_resources.track_performance
        else None
      )

      _run_consumer(conf, result_tensor, shared_resources)

      file_durations, perf_result = _cleanup_processes(
        file_analyzer_proc,
        producer_processes,
        worker_processes,
        perf_tracking_resources,
        shared_resources,
        processing_state,
      )

    stop = time.perf_counter()
    end_timepoint = datetime.now()

    if processing_state.cancel_event.is_set():
      raise RuntimeError(
        f"Analysis was cancelled due to an error. Please check the logs: {logging_resources.log_file.absolute()}"
      )

    result = strategy.create_result(result_tensor, conf, file_paths, file_durations)

    _handle_statistics(
      conf,
      strategy,
      specific_config,
      result,
      processing_state,
      start_time,
      start_timepoint,
      end_timepoint,
      stop,
      start,
      file_durations,
      memory_layout,
      logging_resources,
      perf_result,
    )

    return result

  finally:
    _cleanup_logging(logging_resources)


def _parse_input_files(
  input_files: Path | str | Iterable[Path | str],
) -> OrderedSet[Path]:
  logger = bn_logging.get_logger(__name__)
  logger.info("Getting input files...")
  parsed_audio_paths = set()

  if isinstance(input_files, Path | str):
    input_files = (Path(input_files),)

  if isinstance(input_files, Iterable):
    for inp_audio in input_files:
      if isinstance(inp_audio, Path | str):
        inp_path = Path(inp_audio)
        if inp_path.is_file():
          if inp_path.suffix.upper() in SF_FORMATS:
            parsed_audio_paths.add(inp_path.absolute())
          else:
            raise ValueError(
              f"Input file '{inp_path}' is not a supported audio format! Supported formats: {sorted(SF_FORMATS)}."
            )
        elif inp_path.is_dir():
          parsed_audio_paths.update(get_supported_audio_files(inp_path))
        else:
          raise ValueError(f"Input path '{inp_path}' was not found.")
      else:
        raise ValueError(f"Unsupported input type: {type(inp_audio)}")
  else:
    raise ValueError(f"Unsupported input type: {type(input_files)}")

  file_paths: OrderedSet[Path] = OrderedSet(sorted(set(parsed_audio_paths)))
  logger.info(f"Got {len(file_paths)} audio files for analysis.")

  return file_paths


def _start_file_analyzer(
  config: PredictionConfig,
  file_paths: OrderedSet[Path],
  memory_layout: MemoryLayout,
  shared_resources: SharedResources,
  processing_state: ProcessingState,
) -> threading.Thread:
  """Startet File Analyzer Thread"""
  file_analyzer_proc = threading.Thread(
    target=FilesAnalyzer(
      files=file_paths,
      logging_level=shared_resources.logging_level,
      logging_queue=shared_resources.logging_queue,
      segment_duration_s=config.model_conf.segment_size_s,
      overlap_duration_s=config.processing_conf.overlap_duration_s,
      max_segment_idx_ptr=memory_layout.max_segment_idx_ptr,
      rf_segment_indices=memory_layout.rf_segment_indices,
      analyzing_result=processing_state.analyzer_queue,
      tot_n_segments=processing_state.tot_n_segments_ptr,
      cancel_event=shared_resources.cancel_event,
    ),
    name="FileAnalyzer",
    daemon=True,
  )
  file_analyzer_proc.start()
  return file_analyzer_proc


def _start_producers(
  config: PredictionConfig,
  memory_layout: MemoryLayout,
  shared_resources: SharedResources,
) -> list[mp.Process]:
  """Startet Producer-Prozesse"""
  producer_processes = [
    mp.Process(
      target=ChildProducer(
        files_queue=shared_resources.files_queue,
        batch_size=config.processing_conf.batch_size,
        prd_all_done_event=shared_resources.prd_all_done_event,
        n_slots=shared_resources.n_slots,
        prd_ring_access_lock=shared_resources.prd_ring_access_lock,
        track_performance=shared_resources.track_performance,
        prod_stats_queue=shared_resources.prd_stats_queue,
        rf_file_indices=shared_resources.rf_file_indices,
        rf_segment_indices=shared_resources.rf_segment_indices,
        rf_audio_samples=shared_resources.rf_audio_samples,
        rf_batch_sizes=shared_resources.rf_batch_sizes,
        rf_flags=shared_resources.rf_flags,
        logging_queue=shared_resources.logging_queue,
        logging_level=shared_resources.logging_level,
        sem_free_slots=shared_resources.sem_free_slots,
        sem_filled_slots=shared_resources.sem_filled_slots,
        segment_duration_s=config.model_conf.segment_size_s,
        overlap_duration_s=config.processing_conf.overlap_duration_s,
        target_sample_rate=config.model_conf.sample_rate,
        use_bandpass=config.filtering_conf.use_bandpass,
        bandpass_fmax=config.filtering_conf.bandpass_fmax,
        bandpass_fmin=config.filtering_conf.bandpass_fmin,
        fmin=config.model_conf.sig_fmin,
        fmax=config.model_conf.sig_fmax,
        max_segment_idx_ptr=memory_layout.max_segment_idx_ptr,
        prod_done_ptr=shared_resources.prod_done_ptr,
        n_feeders=shared_resources.n_feeders,
        cancel_event=shared_resources.cancel_event,
      ),
      name=f"ChildProducer-{i}",
      daemon=True,
    )
    for i in range(shared_resources.n_feeders)
  ]

  for p in producer_processes:
    p.start()

  return producer_processes


def _start_workers(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
  shared_resources: SharedResources,
) -> list[mp.Process]:
  devices = (
    config.processing_conf.device
    if isinstance(config.processing_conf.device, list)
    else [config.processing_conf.device] * config.processing_conf.workers
  )

  backend_loader = _create_backend_loader(config)
  worker_processes = strategy.create_workers(
    config,
    specific_config,
    devices,
    backend_loader,
    shared_resources,
  )
  for w in worker_processes:
    w.start()
  return worker_processes


def _run_consumer(
  config: PredictionConfig, result_tensor: TensorBase, shared_resources: SharedResources
) -> None:
  consumer = Consumer(
    n_workers=config.processing_conf.workers,
    worker_queue=shared_resources.worker_queue,
    tensor=result_tensor,
    cancel_event=shared_resources.cancel_event,
  )
  consumer()


def _create_backend_loader(config: PredictionConfig) -> InferenceBackendLoader:
  """Erstellt Backend Loader"""
  if config.model_conf.backend == MODEL_BACKEND_TF:
    backend_type = TFInferenceBackend
  elif config.model_conf.backend == MODEL_BACKEND_PB:
    backend_type = PBInferenceBackend
  else:
    raise AssertionError(f"Unknown backend: {config.model_conf.backend}")

  backend_loader = InferenceBackendLoader(
    model_path=config.model_conf.path,
    backend_type=backend_type,
    backend_kwargs=config.model_conf.backend_kwargs,
  )

  try:
    backend_loader.on_before_worker_initialized()
  except Exception as exc:
    raise RuntimeError(f"Error during backend initialization: {exc}")

  return backend_loader


def _cleanup_processes(
  file_analyzer_proc: threading.Thread,
  producer_processes: list[mp.Process],
  worker_processes: list[mp.Process],
  perf_tracking_resources: PerformanceTrackingResources | None,
  shared_resources: SharedResources,
  processing_state: ProcessingState,
) -> tuple[np.ndarray, PerformanceTrackingResult | None]:
  logger = bn_logging.get_logger(__name__)

  processing_state.processing_finished_event.set()

  file_durations = np.array(
    cast(list[float], processing_state.analyzer_queue.get()), dtype=np.float16
  )
  processing_state.analyzer_queue.close()
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
  if shared_resources.track_performance:
    assert perf_tracking_resources is not None
    perf_tracking_resources.perf_stop_event.set()
    perf_result = cast(
      PerformanceTrackingResult, perf_tracking_resources.perf_res_queue.get()
    )
    perf_tracking_resources.process.join()
    logger.debug("Performance tracker finished.")

  return file_durations, perf_result


def _handle_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
  result: ResultType,
  processing_state: ProcessingState,
  start_time: float,
  start_timepoint: datetime,
  end_timepoint: datetime,
  stop: float,
  start: float,
  file_durations: np.ndarray,
  memory_layout: MemoryLayout,
  logging_resources: LoggingResources,
  perf_result: PerformanceTrackingResult | None = None,
) -> None:
  wall_time_s = stop - start

  if config.output_conf.show_stats in ("minimal", "progress"):
    _show_minimal_statistics(
      config,
      strategy,
      specific_config,
      result,
      processing_state,
      start_timepoint,
      end_timepoint,
      wall_time_s,
      file_durations,
      memory_layout,
    )
  elif config.output_conf.show_stats == "benchmark":
    assert perf_result is not None
    _create_benchmark_statistics(
      config,
      strategy,
      specific_config,
      result,
      processing_state,
      start_time,
      start_timepoint,
      end_timepoint,
      wall_time_s,
      file_durations,
      memory_layout,
      logging_resources,
      perf_result,
    )


def _show_minimal_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
  result: ResultType,
  processing_state: ProcessingState,
  start_timepoint: datetime,
  end_timepoint: datetime,
  wall_time_s: float,
  file_durations: np.ndarray,
  memory_layout: MemoryLayout,
) -> None:
  bmm = strategy.create_minimal_benchmark_meta(
    config,
    specific_config,
    result,
    processing_state,
    start_timepoint,
    end_timepoint,
    wall_time_s,
    file_durations,
    memory_layout,
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
  specific_config: ConfigType,
  result: ResultType,
  processing_state: ProcessingState,
  start_time: float,
  start_timepoint: datetime,
  end_timepoint: datetime,
  wall_time_s: float,
  file_durations: np.ndarray,
  memory_layout: MemoryLayout,
  logging_resources: LoggingResources,
  perf_result: PerformanceTrackingResult,
) -> None:
  """Zeigt vollständige Benchmark-Statistiken an"""
  iso_time = start_timepoint.strftime("%Y%m%dT%H%M%S")

  # Benchmark-Metadaten von Strategy erstellen lassen
  bmm = strategy.create_full_benchmark_meta(
    config,
    specific_config,
    result,
    processing_state,
    start_time,
    start_timepoint,
    end_timepoint,
    wall_time_s,
    file_durations,
    memory_layout,
    perf_result,
  )

  benchmark_dir = logging_resources.benchmark_dir
  benchmark_run_out_dir = logging_resources.benchmark_run_dir

  assert benchmark_dir is not None
  assert benchmark_run_out_dir is not None

  # Dateipfade
  meta_df_out = benchmark_dir / "runs.csv"
  stats_out_json = benchmark_run_out_dir / f"stats-{iso_time}.json"
  stats_human_readable_out = benchmark_run_out_dir / f"stats-{iso_time}.txt"
  result_npz = benchmark_run_out_dir / f"result-{iso_time}.npz"
  result_csv = benchmark_run_out_dir / f"result-{iso_time}.csv"

  # Benchmark-Daten als Dictionary
  bm = asdict(bmm)
  del_keys = [k for k in bm if k.startswith("_")]
  for k in del_keys:
    del bm[k]
  bm = bmm.to_dict()

  # JSON speichern
  with open(stats_out_json, "w", encoding="utf8") as f:
    json.dump(bm, f, indent=2, ensure_ascii=False)

  # CSV-Metadaten speichern
  import pandas as pd

  meta_df = pd.DataFrame.from_records([bm])
  meta_df.to_csv(meta_df_out, mode="a", header=not meta_df_out.exists(), index=False)

  # Human-readable Summary
  n_slots = config.processing_conf.workers + (
    config.processing_conf.workers * config.processing_conf.prefetch_ratio
  )

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
    f"Buffer: {bmm.mem_shm_slots_average_filled:.1f}/{n_slots} filled slots (mean)\n"
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

  # Summary in Datei schreiben
  stats_human_readable_out.write_text(summary, encoding="utf8")

  # Ergebnisse speichern - Strategy entscheidet was gespeichert wird
  saved_files_info = strategy.save_results(result, result_npz, result_csv)

  # Erweiterte Summary mit Dateipfaden
  summary += (
    f"-------------------------------\n"
    f"Benchmark folder:\n"
    f"  {benchmark_run_out_dir.absolute()}\n"
    f"Statistics results written to:\n"
    f"  {stats_human_readable_out.absolute()}\n"
    f"  {stats_out_json.absolute()}\n"
    f"  {meta_df_out.absolute()}\n"
    f"Prediction results written to:\n"
    f"{saved_files_info}"
    f"Log file written to:\n  {logging_resources.log_file.absolute()}\n"
  )

  print(summary)


def _cleanup_logging(logging_resources: LoggingResources) -> None:
  """Cleanup Logging-System"""
  logging_resources.logging_stop_event.set()
  logging_resources.logging_listener.join()
  logging_resources.logging_queue.close()
  logging_resources.logging_queue.join_thread()
  bn_logging.remove_queue_handler(logging_resources.queue_handler)

  # Global log file kopieren
  iso_time = datetime.now().strftime("%Y%m%dT%H%M%S")
  global_log_file = Path(tempfile.gettempdir()) / f"{PKG_NAME}-{iso_time}.log"
  shutil.copyfile(logging_resources.log_file, global_log_file)
