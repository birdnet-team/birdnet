from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.synchronize
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import QueueHandler
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import cast, final

import numpy as np
from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.backends import (
  InferenceBackendLoader,
  PBInferenceBackend,
  TFInferenceBackend,
)
from birdnet.acoustic_models.inference.perf_tracker import PerformanceTrackingResult
from birdnet.acoustic_models.inference_pipeline.configs import (
  PredictionConfig,
)
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_TYPE_ACOUSTIC,
  PKG_NAME,
)
from birdnet.helper import (
  SF_FORMATS,
  RingField,
  get_float_dtype,
  get_max_n_segments,
  get_supported_audio_files,
  is_supported_audio_file,
  uint_ctype_from_dtype,
  uint_dtype_for,
)
from birdnet.local_data import get_benchmark_dir
from birdnet.logging_utils import get_package_logging_level


@dataclass(frozen=True)
class PipelineResources:
  stats_resources: StatisticsResources
  logging_resources: LoggingResources
  processing_state: ProcessingResources
  analyzer_resources: FilesAnalyzerResources
  producer_resources: ProducerResources
  worker_resources: WorkerResources
  ring_buffer_resources: RingBufferResources


class ResourceManager:
  def __init__(self, conf: PredictionConfig, benchmark_dir_name: str):
    self.conf = conf
    self.benchmark_dir_name = benchmark_dir_name
    self.resources: PipelineResources | None = None

  def create_all_resources(self) -> PipelineResources:
    assert self.resources is None
    stats_resources = create_statistics_resources(self.conf, self.benchmark_dir_name)
    logging_resources = create_logging_resources(stats_resources)
    processing_resources = create_processing_resources()
    analyzer_resources = create_analyzer_resources(self.conf)
    producer_resources = create_producer_resources(self.conf, analyzer_resources)
    worker_resources = create_worker_resources(self.conf)
    buf_resources = create_ring_buffer_resources(self.conf, analyzer_resources)

    self.resources = PipelineResources(
      stats_resources=stats_resources,
      logging_resources=logging_resources,
      processing_state=processing_resources,
      analyzer_resources=analyzer_resources,
      producer_resources=producer_resources,
      worker_resources=worker_resources,
      ring_buffer_resources=buf_resources,
    )
    return self.resources


@dataclass(frozen=True)
class RingBufferResources:
  rf_file_indices: RingField
  rf_segment_indices: RingField
  rf_audio_samples: RingField
  rf_batch_sizes: RingField
  rf_flags: RingField
  sem_free_slots: multiprocessing.synchronize.Semaphore
  sem_filled_slots: multiprocessing.synchronize.Semaphore


def create_ring_buffer_resources(
  conf: PredictionConfig, analyzer_resources: FilesAnalyzerResources
) -> RingBufferResources:
  n_slots = conf.processing_conf.n_slots

  rf_file_indices = RingField(
    "bn_ring_file_indices",
    dtype=uint_dtype_for(max(0, analyzer_resources.n_files - 1)),
    shape=(n_slots, conf.processing_conf.batch_size),
  )

  rf_segment_indices = RingField(
    "bn_ring_segment_indices",
    dtype=analyzer_resources.segments_dtype,
    shape=(n_slots, conf.processing_conf.batch_size),
  )

  rf_audio_samples = RingField(
    "bn_ring_audio_samples",
    dtype=np.dtype(np.float32),
    shape=(
      n_slots,
      conf.processing_conf.batch_size,
      conf.model_conf.segment_size_samples,
    ),
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

  return RingBufferResources(
    rf_file_indices=rf_file_indices,
    rf_segment_indices=rf_segment_indices,
    rf_audio_samples=rf_audio_samples,
    rf_batch_sizes=rf_batch_sizes,
    rf_flags=rf_flags,
    sem_free_slots=mp.Semaphore(n_slots),
    sem_filled_slots=mp.Semaphore(0),
  )


@dataclass(frozen=True)
class ProducerResources:
  n_producers: int
  n_finished_pointer: (
    Synchronized[ctypes.c_uint8]
    | Synchronized[ctypes.c_uint16]
    | Synchronized[ctypes.c_uint32]
    | Synchronized[ctypes.c_uint64]
  )
  prd_all_done_event: multiprocessing.synchronize.Event
  ring_access_lock: multiprocessing.synchronize.Lock
  files_queue: mp.Queue


def create_producer_resources(
  conf: PredictionConfig, analyzer_resources: FilesAnalyzerResources
) -> ProducerResources:
  n_producers = min(conf.processing_conf.feeders, analyzer_resources.n_files)
  n_finished_pointer = mp.Value(
    uint_ctype_from_dtype(uint_dtype_for(n_producers)), 0, lock=True
  )

  files_queue = mp.Queue(analyzer_resources.n_files + n_producers)

  for file_idx, file_path in enumerate(analyzer_resources.file_paths):
    files_queue.put((file_idx, file_path), block=False)

  for _ in range(n_producers):
    files_queue.put(None, block=False)

  return ProducerResources(
    n_producers=n_producers,
    n_finished_pointer=n_finished_pointer,
    files_queue=files_queue,
    ring_access_lock=mp.Lock(),
    prd_all_done_event=mp.Event(),
  )


@dataclass(frozen=True)
class WorkerResources:
  results_queue: mp.Queue
  ring_access_lock: multiprocessing.synchronize.Lock
  devices: list[str]
  backend_loader: InferenceBackendLoader


def create_worker_resources(config: PredictionConfig) -> WorkerResources:
  devices = (
    config.processing_conf.device
    if isinstance(config.processing_conf.device, list)
    else [config.processing_conf.device] * config.processing_conf.workers
  )

  backend_loader = _create_backend_loader(config)

  return WorkerResources(
    results_queue=mp.Queue(),
    ring_access_lock=mp.Lock(),
    devices=devices,
    backend_loader=backend_loader,
  )


def _create_backend_loader(config: PredictionConfig) -> InferenceBackendLoader:
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

  return backend_loader


@dataclass(frozen=True)
class FilesAnalyzerResources:
  file_paths: OrderedSet[Path]
  analyzer_queue: mp.Queue
  tot_n_segments_ptr: mp.RawValue
  max_segment_idx_ptr: mp.RawValue
  segments_dtype: np.dtype
  max_segment_defined: bool

  @property
  @final
  def n_files(self) -> int:
    return len(self.file_paths)

  _file_durations: np.ndarray | None = None

  @property
  def file_durations(self) -> np.ndarray | None:
    return self._file_durations

  def collect_file_durations(self) -> np.ndarray:
    durations: list[float] = self.analyzer_queue.get(block=True, timeout=None)
    dtype = get_float_dtype(max(durations))
    file_durations = np.array(durations, dtype=dtype)
    object.__setattr__(self, "_file_durations", file_durations)
    return file_durations


def create_analyzer_resources(conf: PredictionConfig) -> FilesAnalyzerResources:
  reserve_n_segments = 0

  if conf.processing_conf.max_audio_duration_min is not None:
    reserve_n_segments = get_max_n_segments(
      conf.processing_conf.max_audio_duration_min * 60,
      conf.model_conf.segment_size_s,
      conf.processing_conf.overlap_duration_s,
    )

  if reserve_n_segments > 0:
    max_segment_index = reserve_n_segments - 1
    assert max_segment_index >= 0
    segments_dtype = uint_dtype_for(max_segment_index)
    max_segment_ptr_value = max_segment_index
    max_segment_defined = True
  else:
    segments_dtype = np.dtype(np.uint32)
    max_segment_ptr_value = 0
    max_segment_defined = False

  segments_code_type = uint_ctype_from_dtype(segments_dtype)
  max_segment_idx_ptr = mp.RawValue(
    segments_code_type,  # type: ignore
    max_segment_ptr_value,
  )

  logger = bn_logging.get_logger(__name__)
  logger.info(f"Got {len(conf.input_files)} audio files for analysis.")

  return FilesAnalyzerResources(
    analyzer_queue=mp.Queue(),
    tot_n_segments_ptr=mp.RawValue(ctypes.c_uint64, 0),
    max_segment_idx_ptr=max_segment_idx_ptr,
    file_paths=OrderedSet(sorted(conf.input_files)),
    segments_dtype=segments_dtype,
    max_segment_defined=max_segment_defined,
  )


def _parse_input_files(
  input_files: Path | str | Iterable[Path | str],
) -> OrderedSet[Path]:
  logger = bn_logging.get_logger(__name__)
  logger.info("Getting input files...")
  parsed_audio_paths: set[Path] = set()

  if isinstance(input_files, Path | str):
    input_files = (Path(input_files),)

  if isinstance(input_files, Iterable):
    for inp_audio in input_files:
      if isinstance(inp_audio, Path | str):
        inp_path = Path(inp_audio)
        if inp_path.is_file():
          if is_supported_audio_file(inp_path):
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

  for p in parsed_audio_paths:
    assert p.is_absolute()

  file_paths: OrderedSet[Path] = OrderedSet(sorted(set(parsed_audio_paths)))
  logger.info(f"Got {len(file_paths)} audio files for analysis.")

  return file_paths


@dataclass(frozen=True)
class ProcessingResources:
  processing_finished_event: multiprocessing.synchronize.Event
  cancel_event: multiprocessing.synchronize.Event


def create_processing_resources() -> ProcessingResources:
  return ProcessingResources(
    cancel_event=mp.Event(),
    processing_finished_event=mp.Event(),
  )


@dataclass(frozen=True)
class StatisticsResources:
  start: float
  start_time: float
  start_timepoint: datetime

  @property
  def stop(self) -> float | None:
    return self._stop

  @property
  def end_timepoint(self) -> datetime | None:
    return self._end_timepoint

  @property
  def tracking_result(self) -> PerformanceTrackingResult | None:
    return self._tracking_result

  @property
  def start_iso_time(self) -> str:
    return get_iso_time(self.start_timepoint)

  track_performance: bool
  wkr_stats_queue: mp.Queue | None
  prd_stats_queue: mp.Queue | None
  sem_active_workers: multiprocessing.synchronize.Semaphore | None
  perf_res_queue: mp.Queue | None

  benchmarking: bool
  benchmark_dir: Path | None
  benchmark_run_dir: Path | None

  _stop: float | None = None
  _end_timepoint: datetime | None = None
  _tracking_result: PerformanceTrackingResult | None = None

  def mark_stop(self) -> None:
    object.__setattr__(self, "_stop", time.perf_counter())
    object.__setattr__(self, "_end_timepoint", datetime.now())

  def collect_performance_results(self) -> None:
    if self.track_performance:
      assert self.perf_res_queue is not None
      # TODO handle cancel event?
      perf_result = cast(
        PerformanceTrackingResult, self.perf_res_queue.get(block=True, timeout=None)
      )
      object.__setattr__(self, "_tracking_result", perf_result)


def create_statistics_resources(
  conf: PredictionConfig,
  benchmark_dir_name: str,
):
  start = time.perf_counter()
  start_time = time.time()
  start_timepoint = datetime.now()

  track_performance = conf.output_conf.show_stats in ("progress", "benchmark")
  benchmarking = conf.output_conf.show_stats == "benchmark"

  perf_res_queue = None
  wkr_stats_queue = None
  prd_stats_queue = None
  sem_active_workers = None

  if track_performance:
    perf_res_queue = mp.Queue()
    wkr_stats_queue = mp.Queue()
    prd_stats_queue = mp.Queue()
    sem_active_workers = mp.Semaphore(0)

  benchmark_dir = None
  benchmark_run_out_dir = None
  if benchmarking:
    benchmark_dir = get_benchmark_dir(
      model=MODEL_TYPE_ACOUSTIC, dir_name=benchmark_dir_name
    )
    start_iso_time = get_iso_time(start_timepoint)
    benchmark_run_out_dir = benchmark_dir / f"run-{start_iso_time}"
    benchmark_run_out_dir.mkdir(parents=True, exist_ok=True)

  return StatisticsResources(
    start=start,
    start_time=start_time,
    start_timepoint=start_timepoint,
    track_performance=track_performance,
    prd_stats_queue=prd_stats_queue,
    wkr_stats_queue=wkr_stats_queue,
    perf_res_queue=perf_res_queue,
    sem_active_workers=sem_active_workers,
    benchmarking=benchmarking,
    benchmark_dir=benchmark_dir,
    benchmark_run_dir=benchmark_run_out_dir,
  )


def get_iso_time(timepoint: datetime) -> str:
  return timepoint.strftime("%Y%m%dT%H%M%S")


@dataclass(frozen=True)
class LoggingResources:
  log_file: Path
  global_log_file: Path
  logging_level: int
  logging_queue: mp.Queue
  queue_handler: QueueHandler
  stop_logging_event: multiprocessing.synchronize.Event


def create_logging_resources(stats_resources: StatisticsResources) -> LoggingResources:
  if stats_resources.benchmarking:
    assert stats_resources.benchmark_run_dir is not None
    assert stats_resources.start_iso_time is not None

    log_file = (
      stats_resources.benchmark_run_dir / f"log-{stats_resources.start_iso_time}.log"
    )
    log_file.write_text("", encoding="utf-8")
    print(f"Writing logs to: {log_file.absolute()}")
  else:
    log_file = Path(tempfile.gettempdir()) / f"{PKG_NAME}.log"

  global_log_file = (
    Path(tempfile.gettempdir()) / f"{PKG_NAME}-{stats_resources.start_iso_time}.log"
  )

  logging_queue = mp.Queue()
  queue_handler = bn_logging.add_queue_handler(logging_queue)

  return LoggingResources(
    log_file=log_file,
    global_log_file=global_log_file,
    logging_level=get_package_logging_level(),
    logging_queue=logging_queue,
    queue_handler=queue_handler,
    stop_logging_event=mp.Event(),
  )
