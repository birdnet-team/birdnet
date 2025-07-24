from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.synchronize
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import QueueHandler
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.perf_tracker import PerformanceTracker
from birdnet.acoustic_models.inference.pipelining.configs import (
  PredictionConfig,
)
from birdnet.globals import (
  MODEL_TYPE_ACOUSTIC,
  PKG_NAME,
)
from birdnet.helper import (
  RingField,
  get_max_n_segments,
  uint_ctype_from_dtype,
  uint_dtype_for,
)
from birdnet.local_data import get_benchmark_dir
from birdnet.logging_utils import QueueFileWriter, get_package_logging_level


@dataclass
class MemoryLayout:
  n_files: int
  reserve_n_segments: int
  max_segment_idx_ptr: mp.RawValue
  rf_file_indices: RingField
  rf_segment_indices: RingField
  rf_audio_samples: RingField
  rf_batch_sizes: RingField
  rf_flags: RingField


def setup_memory_layout(conf: PredictionConfig, n_files: int) -> MemoryLayout:
  reserve_n_segments = 0
  segments_dtype = np.dtype(np.uint32)

  if conf.processing_conf.max_audio_duration_min is not None:
    reserve_n_segments = get_max_n_segments(
      conf.processing_conf.max_audio_duration_min * 60,
      conf.model_conf.segment_size_s,
      conf.processing_conf.overlap_duration_s,
    )
    segments_dtype = uint_dtype_for(max(0, reserve_n_segments - 1))

  segments_code_type = uint_ctype_from_dtype(segments_dtype)
  max_segment_idx_ptr = mp.RawValue(
    segments_code_type,  # type: ignore
    max(0, reserve_n_segments - 1),
  )

  n_slots = conf.processing_conf.workers + (
    conf.processing_conf.workers * conf.processing_conf.prefetch_ratio
  )

  rf_file_indices = RingField(
    "bn_ring_file_indices",
    dtype=uint_dtype_for(max(0, n_files - 1)),
    shape=(n_slots, conf.processing_conf.batch_size),
  )

  rf_segment_indices = RingField(
    "bn_ring_segment_indices",
    dtype=segments_dtype,
    shape=(n_slots, conf.processing_conf.batch_size),
  )

  model_segment_size_samples = int(
    conf.model_conf.segment_size_s * conf.model_conf.sample_rate
  )
  rf_audio_samples = RingField(
    "bn_ring_audio_samples",
    dtype=np.dtype(np.float32),
    shape=(n_slots, conf.processing_conf.batch_size, model_segment_size_samples),
  )

  rf_batch_sizes = RingField(
    "bn_ring_batch_sizes",
    dtype=uint_dtype_for(conf.processing_conf.batch_size),
    shape=(n_slots,),
  )

  rf_flags = RingField(
    "bn_ring_flags",
    dtype=np.dtype(np.uint8),
    shape=(n_slots,),
  )

  rf_file_indices.cleanup()
  rf_segment_indices.cleanup()
  rf_audio_samples.cleanup()
  rf_batch_sizes.cleanup()
  rf_flags.cleanup()

  return MemoryLayout(
    n_files=n_files,
    reserve_n_segments=reserve_n_segments,
    max_segment_idx_ptr=max_segment_idx_ptr,
    rf_file_indices=rf_file_indices,
    rf_segment_indices=rf_segment_indices,
    rf_audio_samples=rf_audio_samples,
    rf_batch_sizes=rf_batch_sizes,
    rf_flags=rf_flags,
  )


@dataclass
class SharedResources:
  n_slots: int
  n_feeders: int
  model_segment_size_samples: int
  result_dtype: DTypeLike
  sem_free_slots: multiprocessing.synchronize.Semaphore
  sem_filled_slots: multiprocessing.synchronize.Semaphore
  sem_active_workers: multiprocessing.synchronize.Semaphore
  prd_ring_access_lock: multiprocessing.synchronize.Lock
  wkr_ring_access_lock: multiprocessing.synchronize.Lock
  prd_all_done_event: multiprocessing.synchronize.Event
  worker_queue: mp.Queue
  wkr_stats_queue: mp.Queue
  prd_stats_queue: mp.Queue
  files_queue: mp.Queue
  prod_done_ptr: mp.Value
  cancel_event: multiprocessing.synchronize.Event
  logging_queue: mp.Queue
  logging_level: int
  track_performance: bool
  rf_file_indices: RingField
  rf_segment_indices: RingField
  rf_audio_samples: RingField
  rf_batch_sizes: RingField
  rf_flags: RingField


def setup_shared_resources(
  conf: PredictionConfig,
  memory_layout: MemoryLayout,
  logging_resources: LoggingResources,
  processing_state: ProcessingState,
  file_paths: OrderedSet[Path],
) -> SharedResources:
  n_slots = conf.processing_conf.workers + (
    conf.processing_conf.workers * conf.processing_conf.prefetch_ratio
  )
  track_performance = conf.output_conf.show_stats in ("progress", "benchmark")

  model_segment_size_samples = int(
    conf.model_conf.segment_size_s * conf.model_conf.sample_rate
  )
  result_dtype = np.float16 if conf.processing_conf.half_precision else np.float32

  n_feeders = min(conf.processing_conf.feeders, len(file_paths))
  files_queue = mp.Queue(len(file_paths) + n_feeders)
  for file_idx, file_path in enumerate(file_paths):
    files_queue.put((file_idx, file_path), block=False)
  for _ in range(n_feeders):
    files_queue.put(None, block=False)

  prod_done_ptr = mp.Value(
    uint_ctype_from_dtype(uint_dtype_for(n_feeders)),
    0,
    lock=True,
  )

  return SharedResources(
    n_slots=n_slots,
    n_feeders=n_feeders,
    files_queue=files_queue,
    prod_done_ptr=prod_done_ptr,
    model_segment_size_samples=model_segment_size_samples,
    result_dtype=result_dtype,
    sem_free_slots=mp.Semaphore(n_slots),
    sem_filled_slots=mp.Semaphore(0),
    sem_active_workers=mp.Semaphore(0),
    prd_ring_access_lock=mp.Lock(),
    wkr_ring_access_lock=mp.Lock(),
    prd_all_done_event=mp.Event(),
    worker_queue=mp.Queue(),
    wkr_stats_queue=mp.Queue(),
    prd_stats_queue=mp.Queue(),
    cancel_event=processing_state.cancel_event,
    logging_queue=logging_resources.logging_queue,
    logging_level=get_package_logging_level(),
    track_performance=track_performance,
    rf_file_indices=memory_layout.rf_file_indices,
    rf_segment_indices=memory_layout.rf_segment_indices,
    rf_audio_samples=memory_layout.rf_audio_samples,
    rf_batch_sizes=memory_layout.rf_batch_sizes,
    rf_flags=memory_layout.rf_flags,
  )


@dataclass
class ProcessingState:
  analyzer_queue: mp.Queue
  tot_n_segments_ptr: mp.RawValue
  processing_finished_event: multiprocessing.synchronize.Event
  cancel_event: multiprocessing.synchronize.Event


def setup_processing_state() -> ProcessingState:
  """Setup Processing State"""
  return ProcessingState(
    analyzer_queue=mp.Queue(),
    tot_n_segments_ptr=mp.RawValue(ctypes.c_uint64, 0),
    cancel_event=mp.Event(),
    processing_finished_event=mp.Event(),
  )


@dataclass
class PerformanceTrackingResources:
  process: mp.Process
  perf_res_queue: mp.Queue
  perf_stop_event: multiprocessing.synchronize.Event


def start_performance_tracker(
  config: PredictionConfig,
  shared_resources: SharedResources,
  processing_state: ProcessingState,
  start: float,
) -> PerformanceTrackingResources:
  """Startet Performance Tracker"""
  assert shared_resources.track_performance

  perf_res_queue = mp.Queue()
  perf_stop_event = mp.Event()

  perf_tracker = mp.Process(
    target=PerformanceTracker(
      pred_dur_queue=shared_resources.wkr_stats_queue,
      stop_event=perf_stop_event,
      processing_finished_event=processing_state.processing_finished_event,
      update_interval=0.5,
      print_interval=1,
      prod_stats_queue=shared_resources.prd_stats_queue,
      use_stats_from_last_seconds=30,
      n_workers=config.processing_conf.workers,
      start=start,
      sem_filled_slots=shared_resources.sem_filled_slots,
      workers_start=time.perf_counter(),
      segment_size_s=config.model_conf.segment_size_s,
      logging_queue=shared_resources.logging_queue,
      logging_level=shared_resources.logging_level,
      perf_res=perf_res_queue,
      parent_process_id=os.getpid(),
      rf_flags=shared_resources.rf_flags,
      tot_n_segments_ptr=processing_state.tot_n_segments_ptr,
      cancel_event=shared_resources.cancel_event,
      sem_active_workers=shared_resources.sem_active_workers,
    ),
    name="PerformanceTracker",
    daemon=True,
  )
  perf_tracker.start()

  result = PerformanceTrackingResources(
    process=perf_tracker,
    perf_res_queue=perf_res_queue,
    perf_stop_event=perf_stop_event,
  )
  return result


@dataclass
class LoggingResources:
  log_file: Path
  logging_queue: mp.Queue
  logging_listener: threading.Thread
  queue_handler: QueueHandler
  logging_stop_event: multiprocessing.synchronize.Event
  benchmark_dir: Path | None
  benchmark_run_dir: Path | None


def setup_logging(
  conf: PredictionConfig,
  start_timepoint: datetime,
  processing_state: ProcessingState,
  benchmark_dir_name: str,
) -> LoggingResources:
  """Setup Logging-System"""
  iso_time = start_timepoint.strftime("%Y%m%dT%H%M%S")
  log_file = Path(tempfile.gettempdir()) / f"{PKG_NAME}.log"

  benchmark_dir = None
  benchmark_run_out_dir = None
  if conf.output_conf.show_stats == "benchmark":
    benchmark_dir = get_benchmark_dir(
      model=MODEL_TYPE_ACOUSTIC, dir_name=benchmark_dir_name
    )
    benchmark_run_out_dir = benchmark_dir / f"run-{iso_time}"
    benchmark_run_out_dir.mkdir(parents=True, exist_ok=True)
    log_file = benchmark_run_out_dir / f"log-{iso_time}.log"
    print(f"Writing logs to: {log_file.absolute()}")

  logging_stop_event = mp.Event()
  logging_level = get_package_logging_level()
  logging_queue = mp.Queue()

  logging_listener = threading.Thread(
    target=QueueFileWriter(
      log_queue=logging_queue,
      logging_level=logging_level,
      log_file=log_file,
      cancel_event=processing_state.cancel_event,
      stop_event=logging_stop_event,
      processing_finished_event=processing_state.processing_finished_event,
    ),
    name="QueueFileWriter",
    daemon=True,
  )
  logging_listener.start()

  queue_handler = bn_logging.add_queue_handler(logging_queue)

  return LoggingResources(
    log_file=log_file,
    logging_queue=logging_queue,
    logging_listener=logging_listener,
    queue_handler=queue_handler,
    logging_stop_event=logging_stop_event,
    benchmark_dir=benchmark_dir,
    benchmark_run_dir=benchmark_run_out_dir,
  )
