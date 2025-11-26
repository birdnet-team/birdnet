from __future__ import annotations

import inspect
import multiprocessing as mp
import platform
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import psutil

from birdnet.backends import (
  litert_installed,
  tf_installed,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_PRECISIONS,
  MODEL_TYPES,
  NA,
)
from birdnet.local_data import get_package_version


@dataclass
class MinimalBenchmarkMetaBase:
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
  _file_durations: np.ndarray

  @property
  def file_count(self) -> int:
    return len(self._file_durations)

  @property
  def file_duration_sum(self) -> str:
    if self.file_count == 0:
      return NA
    dur_sum = float(np.sum(self._file_durations, dtype=np.float64))
    assert not np.isinf(dur_sum)
    return str(timedelta(seconds=dur_sum))

  @property
  def file_duration_average(self) -> str:
    if self.file_count == 0:
      return NA
    return str(timedelta(seconds=float(self._file_durations.mean())))

  @property
  def file_duration_minimum(self) -> str:
    if self.file_count == 0:
      return NA
    return str(timedelta(seconds=float(self._file_durations.min())))

  @property
  def file_duration_maximum(self) -> str:
    if self.file_count == 0:
      return NA
    return str(timedelta(seconds=float(self._file_durations.max())))

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
      return NA
    result_s = (
      self.file_segments_total * self.model_segment_duration_seconds
    ) / self._time_wall_time_s
    return str(timedelta(seconds=result_s))


@dataclass
class FullBenchmarkMetaBase(MinimalBenchmarkMetaBase):
  @property
  def time_iso(self) -> str:
    return self._start_timepoint.isoformat(timespec="seconds")

  _time_rampup_first_line_s: float

  @property
  def time_rampup_first_line(self) -> str:
    if self._time_rampup_first_line_s is None:
      return NA
    return str(timedelta(seconds=self._time_rampup_first_line_s))

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
    return mp.get_start_method()

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
    return get_package_version()

  @property
  def sw_tf_available(self) -> bool:
    return tf_installed()

  @property
  def sw_litert_available(self) -> bool:
    return litert_installed()

  # Model
  model_type: MODEL_TYPES
  model_backend: str
  model_version: ACOUSTIC_MODEL_VERSIONS
  model_is_custom: bool
  model_path: str
  model_species: int
  model_sig_fmin: int
  model_sig_fmax: int
  model_sample_rate: int
  model_precision: MODEL_PRECISIONS

  file_segments_maximum: int
  file_batches_processed: int

  # Parameter
  param_producers: int
  param_workers: int
  param_overlap_seconds: float
  param_batch_size: int
  param_prefetch_ratio: int
  param_bandpass_fmin: int
  param_bandpass_fmax: int
  param_half_precision: bool
  param_devices: str
  param_inference_library: str | None

  worker_busy_average: float
  worker_wait_time_average_milliseconds: float

  speed_worker_xrt: float

  @property
  def speed_worker_rtf(self) -> float:
    if self.speed_worker_xrt == 0.0:
      return 0.0
    return 1 / self.speed_worker_xrt

  speed_worker_xrt_max: float
  _worker_avg_wall_time_s: float

  @property
  def _speed_worker_rtf_max(self) -> float:
    if self.speed_worker_xrt_max == 0.0:
      return 0.0
    return 1 / self.speed_worker_xrt_max

  @property
  def speed_worker_total_seg_per_second(self) -> float:
    if self.file_segments_total == 0:
      return 0.0
    return self.file_segments_total / self._worker_avg_wall_time_s

  @property
  def speed_worker_total_audio_per_second(self) -> str:
    if self.file_segments_total == 0:
      return NA
    result_s = (
      self.file_segments_total * self.model_segment_duration_seconds
    ) / self._worker_avg_wall_time_s
    return str(timedelta(seconds=result_s))

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

  def to_dict(self) -> dict[str, Any]:
    result = asdict(self)
    del_keys = [k for k in result if k.startswith("_")]
    for k in del_keys:
      del result[k]

    for name, _attr in inspect.getmembers(
      self.__class__, lambda o: isinstance(o, property)
    ):
      if name.startswith("_"):
        continue
      try:
        result[name] = getattr(self, name)
      except Exception as exc:
        result[name] = f"<error: {exc}>"
    # sort result by keys
    result = OrderedDict(sorted(result.items()))
    return result
