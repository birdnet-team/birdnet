from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.synchronize
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import QueueHandler
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Self, cast, final

import numpy as np

from birdnet.acoustic_models.inference.perf_tracker import (
  AcousticProgressStats,
  PerformanceTrackingResult,
)
from birdnet.acoustic_models.inference_pipeline.configs import InferenceConfig
from birdnet.acoustic_models.inference_pipeline.logs import add_session_queue_handler
from birdnet.backends import BackendLoader
from birdnet.base import get_session_id_hash
from birdnet.globals import MODEL_TYPE_ACOUSTIC, PKG_NAME
from birdnet.helper import (
  get_float_dtype,
  get_n_segments_speed,
  get_uint_dtype,
  uint_ctype_from_dtype,
)
from birdnet.local_data import get_benchmark_dir
from birdnet.logging_utils import get_package_logging_level
from birdnet.shm import RingField


@dataclass(frozen=True)
class PipelineResources:
  stats_resources: StatisticsResources
  logging_resources: LoggingResources
  processing_resources: ProcessingResources
  analyzer_resources: FilesAnalyzerResources
  producer_resources: ProducerResources
  worker_resources: WorkerResources
  ring_buffer_resources: RingBufferResources

  def reset(self) -> None:
    self.processing_resources.reset()
    self.analyzer_resources.reset()
    self.producer_resources.reset()
    self.worker_resources.reset()
    self.stats_resources.reset()
    self.logging_resources.reset()
    self.ring_buffer_resources.reset()


class ResourceManager:
  def __init__(self, conf: InferenceConfig) -> None:
    self.conf = conf
    self._resources: PipelineResources | None = None

  def create_resources(
    self, session_id: str, benchmark_dir_name: str
  ) -> PipelineResources:
    assert self._resources is None
    stats_resources = StatisticsResources.create(
      session_id, self.conf, benchmark_dir_name
    )
    logging_resources = LoggingResources.create(session_id, stats_resources)
    processing_resources = ProcessingResources.create()
    analyzer_resources = FilesAnalyzerResources.create(self.conf)
    producer_resources = ProducerResources.create(self.conf)
    worker_resources = WorkerResources.create(self.conf)
    buf_resources = RingBufferResources.create(
      session_id, self.conf, analyzer_resources
    )

    self._resources = PipelineResources(
      stats_resources=stats_resources,
      logging_resources=logging_resources,
      processing_resources=processing_resources,
      analyzer_resources=analyzer_resources,
      producer_resources=producer_resources,
      worker_resources=worker_resources,
      ring_buffer_resources=buf_resources,
    )
    return self.resources

  @property
  @final
  def resources(self) -> PipelineResources:
    assert self._resources is not None
    return self._resources


@dataclass(frozen=True)
class RingBufferResources:
  rf_file_indices: RingField
  rf_segment_indices: RingField
  rf_audio_samples: RingField
  rf_batch_sizes: RingField
  rf_flags: RingField
  sem_free_slots: multiprocessing.synchronize.Semaphore
  sem_filled_slots: multiprocessing.synchronize.Semaphore

  def reset(self) -> None:
    pass

  @classmethod
  def _create(
    cls,
    session_id: str,
    n_slots: int,
    batch_size: int,
    segment_size_samples: int,
    segments_dtype: np.dtype,
    max_n_files: int,
  ) -> Self:
    # session_id is required to run multiple sessions in parallel
    # in multiple processes or threads in the same session

    sid_hash = get_session_id_hash(session_id)

    rf_file_indices = RingField(
      f"bn_file_idx_{sid_hash}",
      dtype=get_uint_dtype(max(0, max_n_files - 1)),
      shape=(n_slots, batch_size),
    )

    rf_segment_indices = RingField(
      f"bn_seg_idx_{sid_hash}",
      dtype=segments_dtype,
      shape=(n_slots, batch_size),
    )

    rf_audio_samples = RingField(
      f"bn_samples_{sid_hash}",
      dtype=np.dtype(np.float32),
      shape=(
        n_slots,
        batch_size,
        segment_size_samples,
      ),
    )

    rf_batch_sizes = RingField(
      f"bn_bs_{sid_hash}",
      dtype=get_uint_dtype(batch_size),
      shape=(n_slots,),
    )

    rf_flags = RingField(
      f"bn_flags_{sid_hash}",
      dtype=np.dtype(np.uint8),
      shape=(n_slots,),
    )

    rf_file_indices.cleanup(sid_hash)
    rf_segment_indices.cleanup(sid_hash)
    rf_audio_samples.cleanup(sid_hash)
    rf_batch_sizes.cleanup(sid_hash)
    rf_flags.cleanup(sid_hash)

    return cls(
      rf_file_indices=rf_file_indices,
      rf_segment_indices=rf_segment_indices,
      rf_audio_samples=rf_audio_samples,
      rf_batch_sizes=rf_batch_sizes,
      rf_flags=rf_flags,
      sem_free_slots=mp.Semaphore(n_slots),
      sem_filled_slots=mp.Semaphore(0),
    )

  @classmethod
  def create(
    cls,
    session_id: str,
    conf: InferenceConfig,
    analyzer_resources: FilesAnalyzerResources,
  ) -> RingBufferResources:
    return cls._create(
      session_id=session_id,
      n_slots=conf.processing_conf.n_slots,
      batch_size=conf.processing_conf.batch_size,
      segment_size_samples=conf.model_conf.segment_size_samples,
      segments_dtype=analyzer_resources.segments_dtype,
      max_n_files=conf.processing_conf.max_n_files,
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
  all_finished: multiprocessing.synchronize.Event
  ring_access_lock: multiprocessing.synchronize.Lock
  input_queue: Queue
  unprocessed_inputs_queue: Queue
  start_signals: list[multiprocessing.synchronize.Event]

  _unprocessed_inputs: set[int] | None = None

  @property
  def unprocessed_inputs(self) -> set[int]:
    assert self._unprocessed_inputs is not None
    return self._unprocessed_inputs

  def reset(self) -> None:
    object.__setattr__(self, "_unprocessed_inputs", None)
    self.n_finished_pointer.value = 0  # type: ignore
    self.all_finished.clear()
    for start_signal in self.start_signals:
      start_signal.clear()

  @classmethod
  def create(cls, conf: InferenceConfig) -> ProducerResources:
    n_producers = conf.processing_conf.feeders
    n_finished_pointer = mp.Value(
      uint_ctype_from_dtype(get_uint_dtype(n_producers)),  # type: ignore
      0,
      lock=True,
    )  # type: ignore

    return ProducerResources(
      n_producers=n_producers,
      n_finished_pointer=n_finished_pointer,
      input_queue=Queue(),
      ring_access_lock=mp.Lock(),
      all_finished=mp.Event(),
      start_signals=[mp.Event() for _ in range(n_producers)],
      unprocessed_inputs_queue=Queue(),
    )

  def collect_unprocessed_inputs(self) -> None:
    unprocessed_inputs: set[int] = set()
    for _ in range(self.n_producers):
      unprocessed = self.unprocessed_inputs_queue.get(block=True, timeout=None)
      unprocessed_inputs.update(unprocessed)
    object.__setattr__(self, "_unprocessed_inputs", unprocessed_inputs)


@dataclass(frozen=True)
class WorkerResources:
  results_queue: Queue
  ring_access_lock: multiprocessing.synchronize.Lock
  devices: list[str]
  backend_loader: BackendLoader
  start_signals: list[multiprocessing.synchronize.Event]

  @classmethod
  def create(cls, config: InferenceConfig) -> WorkerResources:
    n_workers = config.processing_conf.workers
    devices = (
      config.processing_conf.device
      if isinstance(config.processing_conf.device, list)
      else [config.processing_conf.device] * n_workers
    )

    backend_loader = BackendLoader(
      model_path=config.model_conf.path,
      backend_type=config.model_conf.backend_type,
      backend_kwargs=config.model_conf.backend_kwargs,
    )

    return WorkerResources(
      results_queue=Queue(),
      ring_access_lock=mp.Lock(),
      devices=devices,
      backend_loader=backend_loader,
      start_signals=[mp.Event() for _ in range(n_workers)],
    )

  def reset(self) -> None:
    for start_signal in self.start_signals:
      start_signal.clear()


# def _create_backend_loader(config: PredictionConfig) -> InferenceBackendLoader:
#   if config.model_conf.backend == MODEL_BACKEND_TF:
#     backend_type = TFInferenceBackend
#   elif config.model_conf.backend == MODEL_BACKEND_PB:
#     backend_type = PBInferenceBackend
#   else:
#     raise AssertionError(f"Unknown backend: {config.model_conf.backend}")

#   backend_loader = InferenceBackendLoader(
#     model_path=config.model_conf.path,
#     backend_type=backend_type,
#     backend_kwargs=config.model_conf.backend_kwargs,
#   )

#   return backend_loader


@dataclass(frozen=True)
class FilesAnalyzerResources:
  input_queue: queue.Queue
  analyzer_queue: queue.Queue
  tot_n_segments_ptr: mp.RawValue  # type: ignore
  max_segment_idx_ptr: mp.RawValue  # type: ignore
  max_segment_idx_init_value: int
  finished: threading.Event
  # each resource needs own start signal to allow resetting it individually
  start_signal: threading.Event
  segments_dtype: np.dtype

  _unprocessed_inputs: set[int] | None = None

  _input_durations: np.ndarray | None = None

  @property
  def input_durations(self) -> np.ndarray:
    assert self._input_durations is not None
    return self._input_durations

  @property
  def unprocessed_inputs(self) -> set[int]:
    assert self._unprocessed_inputs is not None
    return self._unprocessed_inputs

  def collect_input_durations(self) -> None:
    durations: list[float] = self.analyzer_queue.get(block=True, timeout=None)
    dtype = get_float_dtype(max(durations))
    file_durations = np.array(durations, dtype=dtype)
    object.__setattr__(self, "_input_durations", file_durations)

  def reset(self) -> None:
    object.__setattr__(self, "_input_durations", None)
    object.__setattr__(self, "_unprocessed_inputs", None)
    self.tot_n_segments_ptr.value = 0
    self.max_segment_idx_ptr.value = self.max_segment_idx_init_value
    self.finished.clear()
    self.start_signal.clear()

  @classmethod
  def create(cls, conf: InferenceConfig) -> FilesAnalyzerResources:
    reserve_n_segments = 0

    if conf.processing_conf.max_audio_duration_min is not None:
      reserve_n_segments = get_n_segments_speed(
        conf.processing_conf.max_audio_duration_min * 60,
        conf.model_conf.segment_size_s,
        conf.processing_conf.overlap_duration_s,
        conf.processing_conf.speed,
      )

    if reserve_n_segments > 0:
      max_segment_index = reserve_n_segments - 1
      assert max_segment_index >= 0
      segments_dtype = get_uint_dtype(max_segment_index)
      max_segment_ptr_value = max_segment_index
    else:
      segments_dtype = np.dtype(np.uint32)
      max_segment_ptr_value = 0

    segments_code_type = uint_ctype_from_dtype(segments_dtype)
    max_segment_idx_ptr = mp.RawValue(
      segments_code_type,  # type: ignore
      max_segment_ptr_value,
    )

    return FilesAnalyzerResources(
      analyzer_queue=queue.Queue(),
      input_queue=queue.Queue(),
      tot_n_segments_ptr=mp.RawValue(ctypes.c_uint64, 0),
      max_segment_idx_ptr=max_segment_idx_ptr,
      segments_dtype=segments_dtype,
      max_segment_idx_init_value=max_segment_ptr_value,
      finished=threading.Event(),
      start_signal=threading.Event(),
    )


@dataclass(frozen=True)
class ProcessingResources:
  processing_finished_event: multiprocessing.synchronize.Event
  cancel_event: multiprocessing.synchronize.Event

  # end listening for new files
  end_event: multiprocessing.synchronize.Event

  current_run_nr: int

  @property
  def is_first_run(self) -> bool:
    return self.current_run_nr == 1

  def reset(self) -> None:
    self.processing_finished_event.clear()
    self.cancel_event.clear()
    self.end_event.clear()

  def increment_run_nr(self) -> None:
    object.__setattr__(self, "current_run_nr", self.current_run_nr + 1)

  @classmethod
  def create(cls) -> ProcessingResources:
    return ProcessingResources(
      cancel_event=mp.Event(),
      processing_finished_event=mp.Event(),
      end_event=mp.Event(),
      current_run_nr=1,
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
  wkr_stats_queue: Queue | None
  prd_stats_queue: Queue | None
  sem_active_workers: multiprocessing.synchronize.Semaphore | None
  perf_res_queue: Queue | None
  perf_res_start_signal: multiprocessing.synchronize.Event | None

  use_callback: bool
  callback_fn: Callable[[AcousticProgressStats], None] | None
  callback_queue: Queue | None
  callback_start_signal: threading.Event | None

  benchmarking: bool
  benchmark_dir: Path | None
  benchmark_session_dir: Path | None
  benchmark_dir_name: str

  _stop: float | None = None
  _end_timepoint: datetime | None = None
  _tracking_result: PerformanceTrackingResult | None = None

  def save_end_time(self) -> None:
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

  @classmethod
  def create(
    cls,
    session_id: str,
    conf: InferenceConfig,
    benchmark_dir_name: str,
  ) -> StatisticsResources:
    start = time.perf_counter()
    start_time = time.time()
    start_timepoint = datetime.now()

    track_performance = conf.output_conf.show_stats in ("progress", "benchmark")
    benchmarking = conf.output_conf.show_stats == "benchmark"
    use_callback = track_performance and conf.output_conf.progress_callback is not None

    perf_res_queue = None
    perf_res_start_signal = None
    wkr_stats_queue = None
    prd_stats_queue = None
    sem_active_workers = None

    if track_performance:
      perf_res_queue = Queue()
      perf_res_start_signal = mp.Event()
      wkr_stats_queue = Queue()
      prd_stats_queue = Queue()
      sem_active_workers = mp.Semaphore(0)

    callback_start_signal = None
    callback_queue = None
    callback_fn = None

    if use_callback:
      callback_start_signal = threading.Event()
      callback_queue = Queue()
      callback_fn = conf.output_conf.progress_callback

    benchmark_dir = None
    benchmark_session_dir = None
    if benchmarking:
      benchmark_dir = get_benchmark_dir(
        model=MODEL_TYPE_ACOUSTIC, dir_name=benchmark_dir_name
      )
      session_id_hash = get_session_id_hash(session_id)
      start_iso_time = get_iso_time(start_timepoint)
      benchmark_session_dir = (
        benchmark_dir / f"{start_iso_time}-session-{session_id_hash}"
      )
      benchmark_session_dir.mkdir(parents=True, exist_ok=True)

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
      benchmark_session_dir=benchmark_session_dir,
      perf_res_start_signal=perf_res_start_signal,
      benchmark_dir_name=benchmark_dir_name,
      use_callback=use_callback,
      callback_queue=callback_queue,
      callback_start_signal=callback_start_signal,
      callback_fn=callback_fn,
    )

  def reset(self) -> None:
    start = time.perf_counter()
    start_time = time.time()
    start_timepoint = datetime.now()

    object.__setattr__(self, "_stop", None)
    object.__setattr__(self, "_end_timepoint", None)
    object.__setattr__(self, "_tracking_result", None)
    object.__setattr__(self, "start", start)
    object.__setattr__(self, "start_time", start_time)
    object.__setattr__(self, "start_timepoint", start_timepoint)


def get_iso_time(timepoint: datetime) -> str:
  return timepoint.strftime("%Y%m%dT%H%M%S")


@dataclass(frozen=True)
class LoggingResources:
  session_log_file: Path
  global_log_file: Path
  logging_level: int
  logging_queue: Queue
  queue_handler: QueueHandler
  stop_logging_event: threading.Event

  def reset(self) -> None:
    pass

  @classmethod
  def create(
    cls, session_id: str, stats_resources: StatisticsResources
  ) -> LoggingResources:
    session_id_hash = get_session_id_hash(session_id)
    if stats_resources.benchmarking:
      assert stats_resources.benchmark_session_dir is not None
      assert stats_resources.start_iso_time is not None

      session_log_file = (
        stats_resources.benchmark_session_dir / f"{session_id_hash}.log"
      )
      session_log_file.write_text("", encoding="utf-8")
      print(f"Writing logs to: {session_log_file.absolute()}")
    else:
      session_log_file = (
        Path(tempfile.gettempdir()) / f"{PKG_NAME}_session_{session_id_hash}.log"
      )

    global_log_file = (
      Path(tempfile.gettempdir()) / f"{PKG_NAME}_session_{session_id}.log"
    )

    logging_queue = Queue()
    queue_handler = add_session_queue_handler(session_id, logging_queue)

    return LoggingResources(
      session_log_file=session_log_file,
      global_log_file=global_log_file,
      logging_level=get_package_logging_level(),
      logging_queue=logging_queue,
      queue_handler=queue_handler,
      stop_logging_event=threading.Event(),
    )
