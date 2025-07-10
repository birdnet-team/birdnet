# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import ctypes
import importlib.metadata
import inspect
import json
import multiprocessing
import multiprocessing as mp
import os
import platform
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

# You'll need these imports in your own code
from datetime import datetime, timedelta
from pathlib import Path

# Next two import lines for this demo only
# backend_protocol.py
from typing import Any, Dict, Literal, final

import numpy as np
import psutil
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.files_analyzer import (
  FilesAnalyzer,
  FilesAnalyzerMeta,
)
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTracker,
  PerformanceTrackingResult,
)

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.acoustic_models.inference.producer import ChildProducer
from birdnet.acoustic_models.inference.species_tensor import SpeciesTensor
from birdnet.acoustic_models.inference.worker import ChildWorker
from birdnet.base import (
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPES,
  MODEL_VERSION_V2_4,
  MODEL_VERSIONS,
)
from birdnet.globals import PKG_NAME, WRITABLE_FLAG
from birdnet.helper import (
  RingField,
  create_shm_ring,
  get_max_n_segments,
  get_supported_audio_files,
  uint_ctype_from_dtype,
  uint_dtype_for,
)
from birdnet.io_lock import IOLockHandler
from birdnet.local_data import get_benchmark_dir
from birdnet.logging_utils import QueueFileWriter, get_package_logging_level


@dataclass
class MinimalBenchmarkMeta:
  # Timestamp
  _start_timepoint: datetime
  _end_timepoint: datetime

  _time_wall_time_s: float

  @property
  def time_begin(self) -> str:
    return self._start_timepoint.strftime("%m/%d/%Y %I:%M %p")

  @property
  def time_end(self) -> str:
    return self._end_timepoint.strftime("%m/%d/%Y %I:%M %p")

  @property
  def time_wall_time(self) -> str:
    return str(timedelta(seconds=self._time_wall_time_s))

  # Dataset
  file_count: int
  _file_durations_total: float
  _file_durations_average: float
  _file_durations_minimum: float
  _file_durations_maximum: float

  @property
  def file_duration_sum(self) -> str:
    if self.file_count == 0:
      return "N/A"
    return str(timedelta(seconds=self._file_durations_total))

  @property
  def file_duration_average(self) -> str:
    if self.file_count == 0:
      return "N/A"
    return str(timedelta(seconds=self._file_durations_average))

  @property
  def file_duration_minimum(self) -> str:
    if self.file_count == 0:
      return "N/A"
    return str(timedelta(seconds=self._file_durations_minimum))

  @property
  def file_duration_maximum(self) -> str:
    if self.file_count == 0:
      return "N/A"
    return str(timedelta(seconds=self._file_durations_maximum))

  file_formats: str

  # Memory
  mem_result_total_memory_usage_MiB: float

  mem_shm_size_file_indices_MiB: float
  mem_shm_size_segment_indices_MiB: float
  mem_shm_size_audio_samples_MiB: float
  mem_shm_size_batch_sizes_MiB: float
  mem_shm_size_flags_MiB: float

  @property
  def mem_shm_size_total_MiB(self) -> float:
    return (
      self.mem_shm_size_file_indices_MiB
      + self.mem_shm_size_segment_indices_MiB
      + self.mem_shm_size_audio_samples_MiB
      + self.mem_shm_size_batch_sizes_MiB
      + self.mem_shm_size_flags_MiB
    )

  # Speed

  file_segments_total: int
  model_segment_duration_seconds: float

  @property
  def speed_total_rtf(self) -> float:
    if self.file_segments_total == 0:
      return 0.0
    return self._time_wall_time_s / (
      self.file_segments_total * self.model_segment_duration_seconds
    )

  @property
  def speed_total_xrt(self) -> float:
    if self.speed_total_rtf == 0.0:
      return 0.0
    return 1 / self.speed_total_rtf

  @property
  def speed_total_seg_per_second(self) -> float:
    if self.file_segments_total == 0:
      return 0.0
    return self.file_segments_total / self._time_wall_time_s

  @property
  def speed_total_audio_per_second(self) -> str:
    if self.file_segments_total == 0:
      return "N/A"
    result_s = (
      self.file_segments_total * self.model_segment_duration_seconds
    ) / self._time_wall_time_s
    return str(timedelta(seconds=result_s))


@dataclass
class FullBenchmarkMeta(MinimalBenchmarkMeta):
  @property
  def time_iso(self) -> str:
    return self._start_timepoint.isoformat(timespec="seconds")

  _time_rampup_first_line_s: float
  _time_rampup_first_prediction_s: float | None

  @property
  def time_rampup_first_line(self) -> str:
    if self._time_rampup_first_line_s is None:
      return "N/A"
    return str(timedelta(seconds=self._time_rampup_first_line_s))

  @property
  def time_rampup_first_prediction(self) -> str:
    if self._time_rampup_first_prediction_s is None:
      return "N/A"
    return str(timedelta(seconds=self._time_rampup_first_prediction_s))

  # Hardware
  @property
  def hw_host(self) -> str:
    return platform.node()

  @property
  def hw_cpu(self) -> str:
    return platform.processor()

  @property
  def hw_cpu_physical_cores(self) -> int:
    return psutil.cpu_count(logical=False) or -1

  @property
  def hw_cpu_logical_cores(self) -> int:
    return psutil.cpu_count(logical=True) or -1

  @property
  def hw_ram_GiB(self) -> float:
    return psutil.virtual_memory().total / 1024**3

  @property
  def sw_start_method(self) -> str:
    return multiprocessing.get_start_method()

  # Software
  @property
  def sw_os(self) -> str:
    return f"{platform.system()} {platform.release()}"

  @property
  def sw_python_version(self) -> str:
    return platform.python_version()

  @property
  def sw_python_implementation(self) -> str:
    return platform.python_implementation()

  @property
  def sw_package_version(self) -> str:
    return importlib.metadata.version(PKG_NAME)

  # Model
  model_type: str
  model_backend: str
  model_version: str
  model_is_custom: bool
  model_path: str
  model_species: int
  model_sig_fmin: int
  model_sig_fmax: int
  model_sample_rate: int

  file_segments_maximum: int
  file_batches_processed: int

  # Parameter
  param_producers: int
  param_workers: int
  param_overlap_seconds: float
  param_batch_size: int
  param_top_k: int
  param_prefetch_ratio: int
  param_sigmoid_apply: bool
  param_sigmoid_sensitivity: float | None
  param_bandpass_use: bool
  param_bandpass_fmin: int | None
  param_bandpass_fmax: int | None
  param_half_precision: bool
  param_confidence_threshold_default: float | None
  param_confidence_threshold_custom: int
  param_custom_species: int
  param_devices: str

  worker_busy_average: float
  worker_wait_time_average_milliseconds: float

  speed_worker_xrt: float

  @property
  def speed_worker_rtf(self) -> float:
    if self.speed_worker_xrt == 0.0:
      return 0.0
    return 1 / self.speed_worker_xrt

  speed_worker_xrt_max: float

  @property
  def _speed_worker_rtf_max(self) -> float:
    if self.speed_worker_xrt_max == 0.0:
      return 0.0
    return 1 / self.speed_worker_xrt_max

  # Memory
  mem_shm_ringsize: int

  mem_memory_usage_maximum_MiB: float
  mem_memory_usage_average_MiB: float
  cpu_usage_maximum_pct: float
  cpu_usage_average_pct: float

  mem_shm_slots_average_free: float

  @property
  def mem_shm_slots_average_filled(self) -> float:
    return self.mem_shm_ringsize - self.mem_shm_slots_average_free

  mem_shm_slots_average_busy: float
  mem_shm_slots_average_buffered: float

  # avg_free_slots_last: float
  # avg_filled_slots_last: float
  # avg_busy_slots_last: float
  # avg_preloaded_slots_last: float
  # avg_busy_workers_last: float

  # --- automatische Serialisierung ----------------------------------
  def to_dict(self) -> dict[str, Any]:
    result = asdict(self)  # Dataclass-Felder
    del_keys = [k for k in result if k.startswith("_")]
    for k in del_keys:
      del result[k]

    # Alle Attribute der Klasse durchgehen, die ein property-Objekt sind …
    for name, attr in inspect.getmembers(
      self.__class__, lambda o: isinstance(o, property)
    ):
      if name.startswith("_"):
        continue
      try:
        result[name] = getattr(self, name)  # Property auswerten
      except Exception as exc:  # falls Property Fehler wirft
        result[name] = f"<error: {exc}>"
    # sort result by keys
    result = OrderedDict(sorted(result.items()))
    return result


class AcousticModelBaseV2_4(AcousticModelBase):
  def __init__(self) -> None:
    super().__init__()
    self._model_path: Path | None = None
    self._species_list: OrderedSet[str] | None = None
    self._use_custom_model: bool | None = None

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @property
  def model_path(self) -> Path:
    assert self._model_path is not None
    return self._model_path

  @property
  def species_list(self) -> OrderedSet[str]:
    assert self._species_list is not None
    return self._species_list

  @property
  def use_custom_model(self) -> bool:
    assert self._use_custom_model is not None
    return self._use_custom_model

  @classmethod
  @final
  def get_version(cls) -> MODEL_VERSIONS:
    return MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_ACOUSTIC

  @classmethod
  @final
  def get_sig_fmin(cls) -> int:
    return 0

  @classmethod
  @final
  def get_sig_fmax(cls) -> int:
    return 15_000

  @classmethod
  @final
  def get_sample_rate(cls) -> int:
    return 48_000

  @classmethod
  @final
  def get_segment_size_s(cls) -> float:
    return 3.0

  @classmethod
  @final
  def get_segment_size_samples(cls) -> int:
    return 144_000  # 3.0 * 48_000

  def analyze(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    top_k: int | None = 5,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["no", "minimal", "progress", "benchmark"] = "progress",
    device: str | list[str] = "CPU",
    serial_io: bool = False,
  ):
    start = time.perf_counter()
    start_time = time.time()
    start_timepoint = datetime.now()

    if not batch_size >= 1:
      raise ValueError(
        "Value for 'batch_size' is invalid! It needs to be larger than or equal to 1."
      )

    # if (
    #   default_confidence_threshold is not None
    #   and not 0 <= default_confidence_threshold < 1.0
    # ):
    #   raise ValueError(
    #     "Value for 'min_confidence' is invalid! It needs to be None or in interval [0.0, 1.0)."
    #   )

    if not feeders >= 1:
      raise ValueError(
        "Value for 'feeders' is invalid! It needs to be larger than or equal to 1."
      )

    if not workers >= 1:
      raise ValueError(
        "Value for 'workers' is invalid! It needs to be larger than or equal to 1."
      )

    if not prefetch_ratio >= 0:
      raise ValueError(
        "Value for 'prefetch_ratio' is invalid! It needs to be larger than or equal to 0."
      )

    if not 0 <= overlap_duration_s < 3:
      raise ValueError(
        "Value for 'overlap_duration_s' is invalid! It needs to be in interval [0.0, 3.0)."
      )

    if apply_sigmoid:
      if sigmoid_sensitivity is None:
        raise ValueError(
          "Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!"
        )
      if not 0.5 <= sigmoid_sensitivity <= 1.5:
        raise ValueError(
          "Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval [0.5, 1.5]."
        )

    if use_bandpass:
      if bandpass_fmin is None:
        raise ValueError(
          "Value for 'bandpass_fmin' is required if 'use_bandpass==True'!"
        )
      if bandpass_fmax is None:
        raise ValueError(
          "Value for 'bandpass_fmax' is required if 'use_bandpass==True'!"
        )

      if bandpass_fmin < 0:
        raise ValueError(
          "Value for 'bandpass_fmin' is invalid! It needs to be larger than zero."
        )

      if bandpass_fmax <= bandpass_fmin:
        raise ValueError(
          "Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'."
        )

    if max_audio_duration_min is not None and not max_audio_duration_min > 0:
      raise ValueError(
        "Value for 'max_audio_duration_min' is invalid! It needs to be either None, or larger than zero."
      )

    if show_stats not in ("no", "minimal", "progress", "benchmark"):
      raise ValueError(
        f"Value for 'show_stats' is invalid! It needs to be one of: 'no', 'minimal', 'progress', or 'benchmark'."
      )

    if isinstance(device, list) and len(device) != workers:
      raise ValueError(
        f"Value for 'device' is invalid! Device should be a name, or a list with a length that should match number of workers ({workers})."
      )

    devices = device if isinstance(device, list) else [device] * workers

    if self.get_backend() == "tf":
      for d in devices:
        if "GPU" in d:
          raise ValueError(
            "Value for 'device' is invalid! GPU devices are not supported for TFLite backend! Please use the 'pb' backend instead."
          )

    if custom_species_list is not None:
      for i, species_name in enumerate(custom_species_list):
        if species_name not in self.species_list:
          raise ValueError(
            f"Value for 'custom_species_list' is invalid! Species '{species_name}' is not in the model's species list!"
          )

    if custom_confidence_thresholds is not None and custom_confidence_thresholds:
      for species_name, threshold in custom_confidence_thresholds.items():
        if species_name not in self.species_list:
          raise ValueError(
            f"Value for 'custom_confidence_thresholds' is invalid! Species '{species_name}' is not in the model's species list!"
          )

    if top_k is not None and top_k > len(self.species_list):
      raise ValueError(
        f"top_k cannot be larger than the number of species ({len(self.species_list)})."
      )

    if top_k is None:
      top_k = len(self.species_list)

    if show_stats == "benchmark":
      print("Starting benchmark...")

    track_performance = show_stats in ("progress", "benchmark")

    io_lock = mp.Lock() if serial_io else None
    io_lock_handler = IOLockHandler(serial_io, io_lock)

    log_file = Path(Path(tempfile.gettempdir()) / f"{PKG_NAME}.log")

    benchmark_dir: Path | None = None
    benchmark_run_out_dir: Path | None = None
    iso_time = start_timepoint.strftime("%Y%m%dT%H%M%S")
    if show_stats == "benchmark":
      benchmark_dir = get_benchmark_dir(
        model=AcousticModelBaseV2_4.get_model_type(),
        version=AcousticModelBaseV2_4.get_version(),
      )

      benchmark_run_out_dir = benchmark_dir / f"run-{iso_time}"
      benchmark_run_out_dir.mkdir(parents=True, exist_ok=True)

      log_file = benchmark_run_out_dir / f"log-{iso_time}.log"
      print(f"Writing logs to: {log_file.absolute()}")

    logging_level = get_package_logging_level()
    logging_queue = multiprocessing.Queue()
    logging_listener = multiprocessing.Process(
      target=QueueFileWriter(logging_queue, logging_level, log_file, io_lock_handler),
      daemon=True,
    )
    logging_listener.start()

    queue_handler = bn_logging.add_queue_handler(logging_queue)

    logger = bn_logging.get_logger(__name__)

    logger.info("Getting input files...")
    parsed_audio_paths = set()
    if isinstance(inp, (Path, str)):
      inp = (Path(inp),)

    if isinstance(inp, Iterable):
      for inp_audio in inp:
        if isinstance(inp_audio, (Path, str)):
          inp_path = Path(inp_audio)
          if inp_path.is_file():
            parsed_audio_paths.add(inp_path.absolute())
          elif inp_path.is_dir():
            parsed_audio_paths.update(get_supported_audio_files(inp_path))
          else:
            raise ValueError(f"Input path '{inp_path}' was not found.")
        else:
          raise ValueError(f"Unsupported input type: {type(inp)}")
    else:
      raise ValueError(f"Unsupported input type: {type(inp)}")

    file_paths: OrderedSet[Path] = OrderedSet(sorted(set(parsed_audio_paths)))
    n_files = len(file_paths)

    logger.info(f"Got {len(file_paths)} audio files for analysis.")

    feeders = min(feeders, n_files)
    logger.debug(f"Using {feeders} producer(s) for {n_files} file(s).")
    logger.info("Starting analysis...")

    species_whitelist: np.ndarray
    if custom_species_list is not None and len(custom_species_list) > 0:
      species_ids_whitelist = np.empty(len(custom_species_list), dtype=int)
      for i, species_name in enumerate(custom_species_list):
        assert species_name in self.species_list
        species_id = self.species_list.index(species_name)
        species_ids_whitelist[i] = species_id

      species_whitelist = np.full(self.n_species, fill_value=False, dtype=bool)
      species_whitelist[species_ids_whitelist] = True
    else:
      species_whitelist = np.full(self.n_species, fill_value=True, dtype=bool)
    species_whitelist.setflags(write=False)

    # Thresholds
    if default_confidence_threshold is None:
      default_confidence_threshold = -np.inf
    thresholds = np.full(self.n_species, default_confidence_threshold, np.float32)

    if custom_confidence_thresholds:
      for species_name, threshold in custom_confidence_thresholds.items():
        assert species_name in self.species_list
        species_id = self.species_list.index(species_name)
        thresholds[species_id] = threshold
    thresholds.setflags(write=False)

    # segments_dtype for max file duration:
    # hopsize 3s & overlap 0s: n-segments ÷ 1200
    # ---
    # uint8 = 255 segments = 0 m 12 s
    # uint16 = 65 535 segments = 54 m 36 s
    # uint32 = 4 294 967 295 segments = 2 485 days = 59 652 h
    reserve_n_segments = 0
    segments_dtype = np.dtype(np.uint32)
    if max_audio_duration_min is not None:
      reserve_n_segments = get_max_n_segments(
        max_audio_duration_min * 60, self.get_segment_size_s(), overlap_duration_s
      )
      segments_dtype = uint_dtype_for(max(0, reserve_n_segments - 1))

    segments_code_type = uint_ctype_from_dtype(segments_dtype)
    max_segment_idx_ptr = mp.RawValue(
      segments_code_type, max(0, reserve_n_segments - 1)
    )  # type: ignore

    prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

    n_species = self.n_species

    n_slots = workers + (workers * prefetch_ratio)

    sem_free_slots = mp.Semaphore(n_slots)
    sem_filled_slots = mp.Semaphore(0)
    sem_active_workers = mp.Semaphore(0)
    logger.debug(f"FILL: {sem_filled_slots}, FREE: {sem_free_slots}")

    rf_file_indices = RingField(
      "bn_ring_file_indices",
      dtype=uint_dtype_for(max(0, n_files - 1)),
      shape=(n_slots, batch_size),
    )

    rf_segment_indices = RingField(
      "bn_ring_segment_indices",
      dtype=segments_dtype,
      shape=(n_slots, batch_size),
    )

    rf_audio_samples = RingField(
      "bn_ring_audio_samples",
      dtype=np.dtype(np.float32),
      shape=(n_slots, batch_size, self.get_segment_size_samples()),
    )

    assert batch_size > 0
    rf_batch_sizes = RingField(
      "bn_ring_batch_sizes",
      dtype=uint_dtype_for(batch_size),
      shape=(n_slots,),
    )

    rf_flags = RingField(
      "bn_ring_flags",
      dtype=np.dtype(np.uint8),  # 4 Values
      shape=(n_slots,),
    )

    rf_file_indices.cleanup()
    rf_segment_indices.cleanup()
    rf_audio_samples.cleanup()
    rf_batch_sizes.cleanup()
    rf_flags.cleanup()

    result = SpeciesTensor(
      n_files,
      n_segments=reserve_n_segments,
      top_k=top_k,
      n_species=n_species,
      prob_dtype=prob_dtype,
      segment_indices_dtype=rf_segment_indices.dtype,
      files_dtype=rf_file_indices.dtype,
    )

    species_blacklist = ~species_whitelist[np.newaxis, :]
    species_blacklist.setflags(write=False)
    species_thresholds = thresholds[np.newaxis, :]
    species_thresholds.setflags(write=False)
    worker_queue = mp.Queue()
    worker_slot_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(n_slots - 1)),  # type: ignore
      0,
      lock=True,  # Lock = false?
    )  # type: ignore
    producer_slot_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(n_slots - 1)),  # type: ignore
      0,
      lock=True,
    )  # type: ignore

    pred_dur_queue = mp.SimpleQueue()
    analyzer_queue = mp.SimpleQueue()
    perf_res_queue: mp.SimpleQueue | None = None
    perf_stop_event = mp.Event()
    cancel_event = mp.Event()
    tot_n_segments_ptr = mp.RawValue(ctypes.c_uint64, 0)
    files_queue = mp.Queue()
    for file_idx, file_path in enumerate(file_paths):
      files_queue.put((file_idx, file_path), block=False)
    for _ in range(feeders):
      files_queue.put(None, block=False)
    prod_done_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(feeders)),  # type: ignore
      0,
      lock=True,
    )  # type: ignore

    with (
      create_shm_ring(rf_file_indices),
      create_shm_ring(rf_segment_indices),
      create_shm_ring(rf_audio_samples),
      create_shm_ring(rf_batch_sizes),
      create_shm_ring(rf_flags) as shm_ring_flags,
    ):
      logger.debug("Shared memory initialized.")

      flags = rf_flags.get_array(shm_ring_flags)
      flags[:] = WRITABLE_FLAG

      file_analyzer_proc = mp.Process(
        target=FilesAnalyzer(
          files=file_paths,
          logging_level=logging_level,
          logging_queue=logging_queue,
          segment_duration_s=AcousticModelBaseV2_4.get_segment_size_s(),
          overlap_duration_s=overlap_duration_s,
          max_segment_idx_ptr=max_segment_idx_ptr,
          rf_segment_indices=rf_segment_indices,
          analyzing_result=analyzer_queue,
          tot_n_segments=tot_n_segments_ptr,
          cancel_event=cancel_event,
          io_lock_handler=io_lock_handler,
        ),
        daemon=True,
      )
      file_analyzer_proc.start()

      producer_processes: list[mp.Process] = [
        mp.Process(
          target=ChildProducer(
            files_queue=files_queue,
            slot_ptr=producer_slot_ptr,
            batch_size=batch_size,
            n_slots=n_slots,
            rf_file_indices=rf_file_indices,
            rf_segment_indices=rf_segment_indices,
            rf_audio_samples=rf_audio_samples,
            rf_batch_sizes=rf_batch_sizes,
            rf_flags=rf_flags,
            logging_queue=logging_queue,
            logging_level=logging_level,
            sem_free_slots=sem_free_slots,
            sem_filled_slots=sem_filled_slots,
            segment_duration_s=AcousticModelBaseV2_4.get_segment_size_s(),
            overlap_duration_s=overlap_duration_s,
            target_sample_rate=AcousticModelBaseV2_4.get_sample_rate(),
            use_bandpass=use_bandpass,
            bandpass_fmax=bandpass_fmax,
            bandpass_fmin=bandpass_fmin,
            fmax=AcousticModelBaseV2_4.get_sig_fmax(),
            fmin=AcousticModelBaseV2_4.get_sig_fmin(),
            max_segment_idx_ptr=max_segment_idx_ptr,
            prod_done_ptr=prod_done_ptr,
            n_prods=feeders,
            cancel_event=cancel_event,
            io_lock_handler=io_lock_handler,
          ),
          daemon=True,
        )
        for _ in range(feeders)
      ]
      for p in producer_processes:
        p.start()

      backend_kwargs = [self.get_backend_args() for _ in range(workers)]

      worker_processes = [
        mp.Process(
          target=ChildWorker(
            backend_type=self.get_backend_type(),
            device=devices[i],
            backend_kwargs=backend_kwargs[i],
            top_k=top_k,
            species_thresholds=species_thresholds,
            species_blacklist=species_blacklist,
            batch_size=batch_size,
            n_slots=n_slots,
            slot_ptr=worker_slot_ptr,
            segment_duration_samples=AcousticModelBaseV2_4.get_segment_size_samples(),
            out_q=worker_queue,
            logging_queue=logging_queue,
            logging_level=logging_level,
            rf_file_indices=rf_file_indices,
            rf_segment_indices=rf_segment_indices,
            rf_audio_samples=rf_audio_samples,
            rf_batch_sizes=rf_batch_sizes,
            rf_flags=rf_flags,
            sem_fill=sem_filled_slots,
            sem_free=sem_free_slots,
            apply_sigmoid=apply_sigmoid,
            prob_dtype=prob_dtype,
            sigmoid_sensitivity=sigmoid_sensitivity,
            pred_dur_queue=pred_dur_queue,
            track_performance=track_performance,
            cancel_event=cancel_event,
            sem_active_workers=sem_active_workers,
            io_lock_handler=io_lock_handler,
          ),
          daemon=True,
        )
        for i in range(workers)
      ]

      worker_start = time.perf_counter()
      for w in worker_processes:
        w.start()

      perf_tracker = None
      if track_performance:
        perf_res_queue = mp.SimpleQueue()
        perf_tracker = mp.Process(
          target=PerformanceTracker(
            pred_dur_queue,
            perf_stop_event,
            update_interval=0.5,
            print_interval=1,
            use_stats_from_last_seconds=30,
            n_workers=workers,
            start=start,
            workers_start=worker_start,
            segment_size_s=AcousticModelBaseV2_4.get_segment_size_s(),
            logging_queue=logging_queue,
            logging_level=logging_level,
            perf_res=perf_res_queue,
            parent_process_id=os.getpid(),
            rf_flags=rf_flags,
            tot_n_segments_ptr=tot_n_segments_ptr,
            cancel_event=cancel_event,
            sem_active_workers=sem_active_workers,
          ),
          daemon=True,
        )
        perf_tracker.start()

      consumer = Consumer(
        n_workers=workers,
        worker_queue=worker_queue,
        species_tensor=result,
        max_segment_index=max_segment_idx_ptr,
        cancel_event=cancel_event,
      )
      consumer()

      file_analyzer_proc.join()
      logger.debug("File analyzer finished.")

      for p in producer_processes:
        p.join()
        logger.debug(f"Producer {p.pid} finished.")
      logger.debug("All producers finished.")

      for w in worker_processes:
        w.join()
        logger.debug(f"Worker {w.pid} finished.")
      logger.debug("All workers finished.")

      stop = time.perf_counter()
      end_timepoint = datetime.now()

      if track_performance:
        assert perf_tracker is not None
        perf_stop_event.set()
        perf_tracker.join()
        logger.debug("Performance tracker finished.")

    if cancel_event.is_set():
      logger.error("Analysis was cancelled due to an error.")
      logging_queue.put_nowait(None)
      logging_listener.join()
      raise RuntimeError(
        f"Analysis was cancelled due to an error. Please check the logs for details: {log_file.absolute()}"
      )

    res = PredictionResult(
      tensor=result,
      files=file_paths,
      segment_duration_s=AcousticModelBaseV2_4.get_segment_size_s(),
      overlap_duration_s=overlap_duration_s,
      species_list=self.species_list,
    )
    del result

    if show_stats in ("minimal", "progress"):
      analyzer_res: FilesAnalyzerMeta = analyzer_queue.get()

      bmm = MinimalBenchmarkMeta(
        _start_timepoint=start_timepoint,
        _end_timepoint=end_timepoint,
        _time_wall_time_s=stop - start,
        file_count=n_files,
        _file_durations_total=analyzer_res.file_sum_durations_s,
        _file_durations_average=analyzer_res.file_mean_durations_s,
        _file_durations_minimum=analyzer_res.file_min_durations_s,
        _file_durations_maximum=analyzer_res.file_max_durations_s,
        mem_result_total_memory_usage_MiB=res.memory_size_mb,
        mem_shm_size_file_indices_MiB=rf_file_indices.nbytes / 1024**2,
        mem_shm_size_segment_indices_MiB=rf_segment_indices.nbytes / 1024**2,
        mem_shm_size_audio_samples_MiB=rf_audio_samples.nbytes / 1024**2,
        mem_shm_size_batch_sizes_MiB=rf_batch_sizes.nbytes / 1024**2,
        mem_shm_size_flags_MiB=rf_flags.nbytes / 1024**2,
        file_segments_total=tot_n_segments_ptr.value,
        model_segment_duration_seconds=AcousticModelBaseV2_4.get_segment_size_s(),
        file_formats=", ".join(sorted({x.suffix[1:].upper() for x in file_paths})),
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
    elif show_stats == "benchmark":
      assert track_performance
      assert perf_res_queue is not None
      perf_result: PerformanceTrackingResult = perf_res_queue.get()

      logger.info("Benchmarking is enabled. Collecting performance data...")
      analyzer_res: FilesAnalyzerMeta = analyzer_queue.get()

      bmm = FullBenchmarkMeta(
        _start_timepoint=start_timepoint,
        _end_timepoint=end_timepoint,
        param_producers=feeders,
        param_workers=workers,
        param_devices=", ".join(device) if isinstance(device, list) else device,
        model_type=AcousticModelBaseV2_4.get_model_type(),
        model_version=AcousticModelBaseV2_4.get_version(),
        model_is_custom=self.use_custom_model,
        model_path=str(self.model_path.absolute()),
        model_species=self.n_species,
        file_count=n_files,
        _file_durations_total=analyzer_res.file_sum_durations_s,
        _file_durations_average=analyzer_res.file_mean_durations_s,
        _file_durations_minimum=analyzer_res.file_min_durations_s,
        _file_durations_maximum=analyzer_res.file_max_durations_s,
        file_segments_maximum=max_segment_idx_ptr.value + 1,
        file_segments_total=tot_n_segments_ptr.value,
        model_segment_duration_seconds=AcousticModelBaseV2_4.get_segment_size_s(),
        param_overlap_seconds=overlap_duration_s,
        param_batch_size=batch_size,
        param_top_k=top_k,
        param_prefetch_ratio=prefetch_ratio,
        mem_shm_ringsize=n_slots,
        param_sigmoid_apply=apply_sigmoid,
        param_sigmoid_sensitivity=sigmoid_sensitivity if apply_sigmoid else None,
        param_bandpass_use=use_bandpass,
        param_bandpass_fmin=bandpass_fmin,
        param_bandpass_fmax=bandpass_fmax,
        param_half_precision=half_precision,
        param_confidence_threshold_default=default_confidence_threshold,
        param_custom_species=len(custom_species_list) if custom_species_list else 0,
        param_confidence_threshold_custom=(
          len(custom_confidence_thresholds) if custom_confidence_thresholds else 0
        ),
        _time_rampup_first_line_s=start_time
        - psutil.Process(os.getpid()).create_time(),
        _time_wall_time_s=stop - start,
        mem_result_total_memory_usage_MiB=res.memory_size_mb,
        mem_shm_size_file_indices_MiB=rf_file_indices.nbytes / 1024**2,
        mem_shm_size_segment_indices_MiB=rf_segment_indices.nbytes / 1024**2,
        mem_shm_size_audio_samples_MiB=rf_audio_samples.nbytes / 1024**2,
        mem_shm_size_batch_sizes_MiB=rf_batch_sizes.nbytes / 1024**2,
        mem_shm_size_flags_MiB=rf_flags.nbytes / 1024**2,
        # n_usage_recordings=perf_result.n_usage_recordings,
        mem_memory_usage_maximum_MiB=perf_result.max_memory_usages_MiB,
        mem_memory_usage_average_MiB=perf_result.avg_memory_usages_MiB,
        cpu_usage_maximum_pct=perf_result.max_cpu_usages_pct,
        cpu_usage_average_pct=perf_result.avg_cpu_usages_pct,
        mem_shm_slots_average_free=perf_result.avg_free_slots,
        mem_shm_slots_average_busy=perf_result.avg_busy_slots,
        mem_shm_slots_average_buffered=perf_result.avg_preloaded_slots,
        worker_busy_average=perf_result.avg_busy_workers,
        # avg_free_slots_last=perf_result.avg_free_slots_last,
        # avg_filled_slots_last=n_slots - perf_result.avg_free_slots_last,
        # avg_busy_slots_last=perf_result.avg_busy_slots_last,
        # avg_preloaded_slots_last=perf_result.avg_preloaded_slots_last,
        # avg_busy_workers_last=perf_result.avg_busy_workers_last,
        _time_rampup_first_prediction_s=perf_result.ramp_up_time_until_first_pred_s,
        file_batches_processed=perf_result.total_batches_processed,
        speed_worker_xrt=perf_result.worker_speed_xrt,
        speed_worker_xrt_max=perf_result.worker_speed_xrt_max,
        model_backend=self.get_backend(),
        model_sample_rate=AcousticModelBaseV2_4.get_sample_rate(),
        model_sig_fmin=AcousticModelBaseV2_4.get_sig_fmin(),
        model_sig_fmax=AcousticModelBaseV2_4.get_sig_fmax(),
        worker_wait_time_average_milliseconds=perf_result.avg_wait_time_ms,
        file_formats=", ".join(sorted({x.suffix[1:].upper() for x in file_paths})),
      )

      # bm = OrderedDict()

      # wall_time_s = stop - start
      # pc_segments_per_s = total_segments_processed / wall_time_s
      # samples_per_second = (
      #   AcousticModelBaseV2_4.get_segment_size_samples() * pc_segments_per_s
      # )

      # bm["model_pred_ms_per_segment"] = None
      # bm["pc_segments_per_s"] = pc_segments_per_s
      # bm["pc_audio_min_per_s"] = (
      #   pc_segments_per_s * AcousticModelBaseV2_4.get_segment_size_s() / 60
      # )
      # bm["pc_s_per_audio_h"] = 60 / bm["pc_audio_min_per_s"]

      # total_segments_processed = perf_result["total_segments_processed"]
      # cpu_time_s = perf_result["summed_prediction_duration_s"]
      # model_pred_ms_per_segment = cpu_time_s / total_segments_processed * 1000

      # raw_segments_per_s = total_segments_processed / (
      #   cpu_time_s / perf_result["avg_busy_slots"]
      # )

      # Metrics

      # bm["raw_segments_per_s"] = raw_segments_per_s
      # bm["raw_min_per_s"] = (
      #   bm["raw_segments_per_s"] * AcousticModelBaseV2_4.get_segment_size_s() / 60
      # )
      # bm["raw_avg_segments_per_s_last"] = perf_result["avg_segments_per_s_last"]
      # bm["raw_avg_raw_min_per_s_last"] = (
      #   bm["raw_avg_segments_per_s_last"] * AcousticModelBaseV2_4.get_segment_size_s() / 60
      # )
      # bm["raw_avg_s_for_one_hour_last"] = 60 / bm["raw_avg_raw_min_per_s_last"]
      # bm["raw_s_for_one_hour"] = 60 / bm["raw_min_per_s"]
      # bm["raw_segments_per_s_max"] = perf_result["max_raw_segments_per_s"]
      # bm["raw_min_per_s_max"] = (
      #   bm["raw_segments_per_s_max"] * AcousticModelBaseV2_4.get_segment_size_s() / 60
      # )
      # bm["raw_s_for_one_hour_max"] = 60 / bm["raw_min_per_s_max"]

      # bm["model_pred_ms_per_segment"] = model_pred_ms_per_segment
      # bm["model_pred_ms_per_batch"] = (
      #   bm["cpu_time_s"] / bm["n_batches_processed"] * 1000
      # )

      # bm["real_time_factor"] = 0
      # bm["speed_x_real_time"] = 0

      bm = asdict(bmm)
      del_keys = [k for k in bm if k.startswith("_")]
      for k in del_keys:
        del bm[k]
      bm = bmm.to_dict()
      # print(bm.items())

      assert benchmark_dir is not None
      assert benchmark_run_out_dir is not None

      meta_df_out = benchmark_dir / "runs.csv"
      stats_out_json = benchmark_run_out_dir / f"stats-{iso_time}.json"
      stats_human_readable_out = benchmark_run_out_dir / f"stats-{iso_time}.txt"
      result_csv = benchmark_run_out_dir / f"result-{iso_time}.csv"
      result_npz = benchmark_run_out_dir / f"result-{iso_time}.npz"

      with open(stats_out_json, "w", encoding="utf8") as f:
        json.dump(bm, f, indent=2, ensure_ascii=False)

      meta_df = pd.DataFrame.from_records([bm])
      meta_df.to_csv(
        meta_df_out, mode="a", header=not meta_df_out.exists(), index=False
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
        # f"\tBusy: {bmm.avg_busy_slots:.1f} slots\n"
        # f"\tPreloaded: {bmm.avg_preloaded_slots:.1f} slots\n"
        # f"\tFree: {bmm.avg_free_slots:.1f} slots\n"
        f"Memory usage:\n"
        f"  Program: {bmm.mem_memory_usage_maximum_MiB:.2f} M (total max)\n"
        f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
        f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
        f"Performance:\n"
        f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
        f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
        f"Worker performance:\n"
        f"  {bmm.speed_worker_xrt:.0f} x real-time (RTF: {bmm.speed_worker_rtf:.8f})\n"
        # f"  {bmm.speed_worker_xrt_max:.0f} x real-time (max)\n"
        # f"\tAudio processing (all): {bmm.pc_audio_min_per_s:.2f} min audio/s ({bmm.pc_s_per_audio_h:.2f} s/h audio; {bmm.pc_segments_per_s:.2f} segments/s)\n"
        # f"\tAudio processing (computation):\n"
        # f"\t\tMean: {bmm.raw_min_per_s:.2f} min audio/s ({bmm.raw_s_for_one_hour:.2f} s/h audio; {bmm.raw_segments_per_s:.2f} segments/s)\n"
        # f"\t\tMean (last 30s): {bmm.raw_avg_raw_min_per_s_last:.2f} min audio/s ({bmm.raw_avg_s_for_one_hour_last:.2f} s/h audio; {bmm.raw_avg_segments_per_s_last:.2f} segments/s)\n"
        # f"\t\tBest: {bmm.raw_min_per_s_max:.2f} min audio/s ({bmm.raw_s_for_one_hour_max:.2f} s/h audio; {bmm.raw_segments_per_s_max:.2f} segments/s)\n"
        # f"\tPrediction speed: {bmm.model_pred_ms_per_segment:.2f} ms/segment ({bmm.model_pred_ms_per_batch:.2f} ms/batch)\n"
      )
      stats_human_readable_out.write_text(summary, encoding="utf8")

      print("Saving result using internal format (.npz)...")
      res.dump(result_npz)
      print("Saving result using CSV format (.csv)...")
      res.to_csv(result_csv, encoding="utf-8", silent=False)

      summary += (
        f"-------------------------------\n"
        f"Benchmark folder:\n"
        f"  {benchmark_run_out_dir.absolute()}\n"
        f"Statistics results written to: {benchmark_run_out_dir.absolute()}\n"
        f"  {stats_human_readable_out.absolute()}\n"
        f"  {stats_out_json.absolute()}\n"
        f"  {meta_df_out.absolute()}\n"
        f"Prediction results written to:\n"
        f"  {result_npz.absolute()}\n"
        f"  {result_csv.absolute()}\n"
        f"Log file written to:\n"
        f"  {log_file.absolute()}\n"
      )
      print(summary)

    logging_queue.put_nowait(None)
    logging_listener.join()
    bn_logging.remove_queue_handler(queue_handler)

    global_log_file_iso = Path(
      Path(tempfile.gettempdir()) / f"{PKG_NAME}-{iso_time}.log"
    )
    shutil.copyfile(log_file, global_log_file_iso)
    return res
