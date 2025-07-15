# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import ctypes
import datetime
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.synchronize import Event, Semaphore

import numpy as np
import psutil

import birdnet.logging_utils as bn_logging
from birdnet.globals import READABLE_FLAG, READING_FLAG, WRITABLE_FLAG
from birdnet.helper import RingField


@dataclass
class PerformanceTrackingResult:
  worker_speed_xrt: float
  worker_speed_xrt_max: float
  worker_avg_wall_time_s: float
  total_segments_processed: int
  total_batches_processed: int
  ramp_up_time_until_first_pred_s: float | None
  n_usage_recordings: int

  max_memory_usages_MiB: float
  avg_memory_usages_MiB: float

  max_cpu_usages_pct: float
  avg_cpu_usages_pct: float

  avg_free_slots: float
  avg_busy_slots: float
  avg_preloaded_slots: float
  avg_busy_workers: float

  avg_wait_time_ms: float
  # avg_pred_dur_last_s: float
  # avg_wait_dur_last_ms: float
  # avg_free_slots_last: float
  # avg_busy_slots_last: float
  # avg_preloaded_slots_last: float
  # avg_busy_workers_last: float

  # max_raw_segments_per_s: float

  # avg_segments_per_s_last: float


class DurationTracker:
  def __init__(self, n_last: int):
    self._n_last = n_last
    self._durations = deque(maxlen=n_last)
    self._total_duration = np.nan
    self._avg_duration = np.nan
    self._min_duration = np.nan
    self._max_duration = np.nan
    self._n_recordings = 0

  def add_duration(self, duration: float) -> None:
    self._durations.append(duration)
    self._min_duration = (
      min(self._min_duration, duration) if self._n_recordings > 0 else duration
    )
    self._max_duration = (
      max(self._max_duration, duration) if self._n_recordings > 0 else duration
    )
    self._total_duration = (
      self._total_duration + duration if self._n_recordings > 0 else duration
    )
    self._avg_duration = (
      (self._avg_duration * self._n_recordings + duration) / (self._n_recordings + 1)
      if self._n_recordings > 0
      else duration
    )

    self._n_recordings += 1

  @property
  def avg_duration(self) -> float:
    return self._avg_duration

  @property
  def min_duration(self) -> float:
    return self._min_duration

  @property
  def max_duration(self) -> float:
    return self._max_duration

  @property
  def total_duration(self) -> float:
    return self._total_duration

  @property
  def n_recordings(self) -> int:
    return self._n_recordings

  @property
  def durations(self) -> deque[float]:
    return self._durations

  @property
  def avg_duration_last(self) -> float:
    if len(self._durations) == 0:
      return np.nan
    return np.mean(self._durations)  # type: ignore


class PerformanceTracker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    pred_dur_queue: mp.Queue,
    prod_stats_queue: mp.Queue,
    stop_event: Event,
    update_interval: float,
    print_interval: float,
    use_stats_from_last_seconds: float,
    n_workers: int,
    start: float,
    workers_start: float,
    logging_queue: mp.Queue,
    logging_level: int,
    perf_res: mp.Queue,
    sem_active_workers: Semaphore,
    segment_size_s: float,
    parent_process_id: int,
    rf_flags: RingField,
    tot_n_segments_ptr: ctypes.c_uint64,
    cancel_event: Event,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    assert update_interval <= print_interval

    self._n_workers = n_workers
    self._workers_start = workers_start
    self._prd_stats_queue = prod_stats_queue
    self._perf_res = perf_res
    self._wkr_stats_queue = pred_dur_queue
    self._n_last = int(1 / update_interval * use_stats_from_last_seconds)
    self._sem_active_workers = sem_active_workers
    # self._n_last = print_last_n
    # self._pred_dur_deque = deque(maxlen=self._n_last)
    # self._batch_sizes_deque = deque(maxlen=self._n_last)
    self._update_every = update_interval
    self._stop_event = stop_event
    self._next_print = time.time()
    self._next_update = self._next_print
    self._summed_prd_flush_duration = 0.0
    self._start = start
    self._segment_size_s = segment_size_s
    self._parent_process_id = parent_process_id
    self._print_every = print_interval
    self._rf_flags = rf_flags
    self._shm_ring_flags: shared_memory.SharedMemory | None = None
    self._ring_flags: np.ndarray | None = None
    self._tot_n_segments_ptr = tot_n_segments_ptr
    self._cancel_event = cancel_event

    self._wkr_wall_times = {}
    self._wkr_total_segments_processed = 0
    self._wkr_1_wait_dur_for_filled_slot_tracker = DurationTracker(self._n_last)
    self._wkr_2_search_dur_for_filled_slot_tracker = DurationTracker(self._n_last)
    self._wkr_3_get_job_dur_tracker = DurationTracker(self._n_last)
    self._wkr_4_inference_dur_tracker = DurationTracker(self._n_last)
    self._wkr_5_add_to_queue_dur_tracker = DurationTracker(self._n_last)
    self._wkr_ramp_up_time_until_first_pred = None

    self._prd_wall_times = {}
    self._prd_total_segments_processed = 0
    self._prd_1_batch_loading_dur_tracker = DurationTracker(self._n_last)
    self._prd_2_wait_dur_free_slot_tracker = DurationTracker(self._n_last)
    self._prd_3_free_slot_search_dur_tracker = DurationTracker(self._n_last)
    self._prd_4_flush_dur_tracker = DurationTracker(self._n_last)

  def _get_worker_stats(self) -> None:
    while not self._wkr_stats_queue.empty():
      (
        worker_pid,
        wall_time,
        dur_wait_for_filled_slot,
        dur_search_for_filled_slot,
        dur_get_job,
        dur_inference,
        dur_add_to_queue,
        batch_size,
      ) = self._wkr_stats_queue.get(block=False)
      self._logger.debug(
        f"PerformanceTracker received prediction duration from worker {worker_pid}: "
        f"wall time: {wall_time:.3f}s, wait for filled slot: {dur_wait_for_filled_slot:.3f}s, find filled slot: {dur_search_for_filled_slot:.3f}s, inference: {dur_inference:.3f}s, add to queue: {dur_add_to_queue}s, batch size: {batch_size}"
      )
      self._wkr_wall_times[worker_pid] = wall_time
      self._wkr_total_segments_processed += batch_size

      self._wkr_1_wait_dur_for_filled_slot_tracker.add_duration(
        dur_wait_for_filled_slot
      )
      self._wkr_2_search_dur_for_filled_slot_tracker.add_duration(
        dur_search_for_filled_slot
      )
      self._wkr_3_get_job_dur_tracker.add_duration(dur_get_job)
      self._wkr_4_inference_dur_tracker.add_duration(dur_inference)
      self._wkr_5_add_to_queue_dur_tracker.add_duration(dur_add_to_queue)

      if self._wkr_ramp_up_time_until_first_pred is None:
        self._wkr_ramp_up_time_until_first_pred = (
          time.perf_counter() - self._start - dur_inference
        )
        self._logger.info(
          f"Rampup time until first prediction: {self._wkr_ramp_up_time_until_first_pred:.2f}s"
        )

  def _get_producer_stats(self) -> None:
    while not self._prd_stats_queue.empty():
      (
        prod_pid,
        process_total_duration,
        batch_loading_duration,
        wait_time_for_free_slot,
        free_slot_search_time,
        flush_duration,
        n,
      ) = self._prd_stats_queue.get(block=False)
      self._logger.debug(
        f"PerformanceTracker received producer stats from producer {prod_pid}: "
        f"process: {process_total_duration:.3f}s, batch loading: {batch_loading_duration:.3f}s, "
        f"wait for free slot: {wait_time_for_free_slot:.3f}s, flush: {flush_duration:.3f}s, n: {n}"
      )
      self._prd_wall_times[prod_pid] = process_total_duration
      self._prd_total_segments_processed += n

      self._prd_1_batch_loading_dur_tracker.add_duration(batch_loading_duration)
      self._prd_2_wait_dur_free_slot_tracker.add_duration(wait_time_for_free_slot)
      self._prd_3_free_slot_search_dur_tracker.add_duration(free_slot_search_time)
      self._prd_4_flush_dur_tracker.add_duration(flush_duration)

  def __call__(self):
    self._init_logging()
    self._shm_ring_flags, self._ring_flags = self._rf_flags.attach_and_get_array()
    wall_time = 0
    parent_process = psutil.Process(self._parent_process_id)
    float_n_records = 0
    float_max_memory_usage = 0
    float_avg_memory_usage = 0
    float_avg_cpu_usage = 0
    float_max_cpu_usage = 0
    float_avg_free_slots = 0
    float_avg_busy_slots = 0
    float_avg_preloaded_slots = 0
    float_avg_busy_workers = 0
    max_raw_segments_per_s = 0
    worker_speed_xrt_max = 0

    cpu_usages = deque(maxlen=self._n_last)
    memory_usages = deque(maxlen=self._n_last)
    free_slots = deque(maxlen=self._n_last)
    busy_slots = deque(maxlen=self._n_last)
    preloaded_slots = deque(maxlen=self._n_last)
    busy_workers = deque(maxlen=self._n_last)

    summed_warm_up = 0

    avg_segments_per_s = deque(maxlen=self._n_last)

    while True:
      if self._cancel_event.is_set():
        self._logger.debug("PerformanceTracker canceled because of cancel event.")
        self._uninit_logging()
        return
      processing_finished = self._stop_event.is_set()
      worker_queue_is_empty = self._wkr_stats_queue.empty()
      producer_queue_is_empty = self._prd_stats_queue.empty()
      if processing_finished:
        if worker_queue_is_empty and producer_queue_is_empty:
          # TODO print again final stats
          break

      self._get_worker_stats()
      self._get_producer_stats()

      now = time.time()

      if now >= self._next_update:
        memory_usage = parent_process.memory_full_info().uss
        for child in parent_process.children(recursive=True):
          try:
            memory_usage += child.memory_full_info().uss
          except psutil.NoSuchProcess:
            continue
          except psutil.AccessDenied:
            continue

        memory_usage_MiB = memory_usage / 1024**2
        memory_usages.append(memory_usage_MiB)

        cpu_usage = psutil.cpu_percent()
        cpu_usages.append(cpu_usage)

        float_avg_memory_usage = (
          float_avg_memory_usage * float_n_records + memory_usage_MiB
        ) / (float_n_records + 1)

        float_max_memory_usage = max(float_max_memory_usage, memory_usage_MiB)

        float_avg_cpu_usage = (float_avg_cpu_usage * float_n_records + cpu_usage) / (
          float_n_records + 1
        )
        float_max_cpu_usage = max(float_max_cpu_usage, cpu_usage)

        c = Counter(self._ring_flags)
        n_free = c.get(WRITABLE_FLAG, 0)
        n_preloaded = c.get(READABLE_FLAG, 0)
        n_busy = c.get(READING_FLAG, 0)

        float_avg_free_slots = (float_avg_free_slots * float_n_records + n_free) / (
          float_n_records + 1
        )

        float_avg_busy_slots = (float_avg_busy_slots * float_n_records + n_busy) / (
          float_n_records + 1
        )
        float_avg_preloaded_slots = (
          float_avg_preloaded_slots * float_n_records + n_preloaded
        ) / (float_n_records + 1)

        n_busy_workers = self._sem_active_workers.get_value()
        float_avg_busy_workers = (
          float_avg_busy_workers * float_n_records + n_busy_workers
        ) / (float_n_records + 1)

        _summed_wkr_duration = sum(self._wkr_wall_times.values())
        wkr_proc_audio_duration_s = (
          self._wkr_total_segments_processed * self._segment_size_s
        )

        wkr_speed_xrt = (
          wkr_proc_audio_duration_s / _summed_wkr_duration * len(self._wkr_wall_times)
          if _summed_wkr_duration > 0
          else 0
        )

        worker_speed_xrt_max = max(worker_speed_xrt_max, wkr_speed_xrt)

        free_slots.append(n_free)
        busy_slots.append(n_busy)
        preloaded_slots.append(n_preloaded)
        busy_workers.append(n_busy_workers)

        float_n_records += 1
        self._next_update = now + self._update_every

      if (
        now >= self._next_print
        and self._wkr_1_wait_dur_for_filled_slot_tracker.n_recordings > 0
      ):
        t = time.perf_counter()
        wall_time = t - self._start
        # perf_duration_workers = t - self._workers_start
        # avg = sum(self._pred_dur_deque) / sum(self._batch_sizes_deque)
        # segments_per_s = self._total_segments_processed / wall_time
        # min_per_s = segments_per_s * self._segment_size_s / 60

        memory_usage = parent_process.memory_full_info().uss
        for child in parent_process.children(recursive=True):
          try:
            memory_usage += child.memory_full_info().uss
          except psutil.NoSuchProcess:
            continue
          except psutil.AccessDenied:
            continue
        memory_usage_MiB = memory_usage / 1024**2

        avg_free_slots = np.mean(free_slots) if free_slots else 0
        avg_busy_workers = np.mean(busy_workers) if busy_workers else 0

        # avg_busy_workers = self._sem_active_workers.get_value()

        # raw_segments_per_s_old = (
        #   self._total_segments_processed
        #   / (self._summed_worker_raw_pred_duration / avg_busy_workers)
        #   if avg_busy_workers > 0
        #   else 0
        # )
        wkr_proc_audio_duration_s = (
          self._wkr_total_segments_processed * self._segment_size_s
        )
        prd_proc_audio_duration_s = (
          self._prd_total_segments_processed * self._segment_size_s
        )

        _summed_wkr_duration = sum(self._wkr_wall_times.values())
        _summed_prd_duration = sum(self._prd_wall_times.values())

        wkr_speed_xrt = (
          wkr_proc_audio_duration_s / _summed_wkr_duration * len(self._wkr_wall_times)
          if _summed_wkr_duration > 0
          else 0
        )
        prd_speed_xrt = (
          prd_proc_audio_duration_s / _summed_prd_duration * len(self._prd_wall_times)
          if _summed_prd_duration > 0
          else 0
        )

        wkr_speed_segments_per_s = (
          self._wkr_total_segments_processed
          / _summed_wkr_duration
          * len(self._wkr_wall_times)
          if _summed_wkr_duration > 0
          else 0
        )
        prd_speed_segments_per_s = (
          self._prd_total_segments_processed
          / _summed_prd_duration
          * len(self._prd_wall_times)
          if _summed_prd_duration > 0
          else 0
        )

        # speed_x_real_time_classic = processed_audio_duration_s / wall_time

        # raw_min_per_s = raw_segments_per_s_old * self._segment_size_s / 60

        # max_raw_segments_per_s = max(max_raw_segments_per_s, raw_segments_per_s_old)

        # avg_segments_per_s.append(raw_segments_per_s_old)

        output_msg_fields = [
          # f"inference speed: {self._summed_raw_pred_duration / self._total_segments_processed * 1000:.0f} ms/segment",
          # f"last {len(self._pred_dur_deque)} predictions: {avg * 1000:.0f} ms/segment",
          # f"RTF: {real_time_factor:.8f}x [{raw_segments_per_s:.0f} segm/s]",
          f"F-SPEED: {prd_speed_xrt:.0f} xRT [{prd_speed_segments_per_s:.0f} seg/s]",
          f"W-SPEED: {wkr_speed_xrt:.0f} xRT [{wkr_speed_segments_per_s:.0f} seg/s]",
          # f"SPEED2: {speed_x_real_time_classic:.0f} xRT [{segments_per_s:.0f} seg/s]",
          # f"{raw_min_per_s:.2f} min/s",
          f"MEM: {memory_usage_MiB:.0f} M",
          # f"CPU usage: {cpu_usage:.1f} %",
          f"BUF: {self._ring_flags.shape[0] - avg_free_slots:.0f}/{self._ring_flags.shape[0]}",
          # f"free: {avg_free_slots:.0f}/{self._ring_flags.shape[0]}",
          f"P-WAIT: {self._prd_1_batch_loading_dur_tracker.avg_duration_last * 1000:.2f} ms",
          f"P-BATCH: {self._prd_2_wait_dur_free_slot_tracker.avg_duration_last * 1000:.2f} ms",
          f"P-SEARCH: {self._prd_3_free_slot_search_dur_tracker.avg_duration_last * 1000:.2f} ms",
          f"P-FLUSH: {self._prd_4_flush_dur_tracker.avg_duration_last * 1000:.2f} ms",
          f"W-WAIT: {self._wkr_1_wait_dur_for_filled_slot_tracker.avg_duration * 1000:.2f} ms",
          f"W-SEARCH: {self._wkr_2_search_dur_for_filled_slot_tracker.avg_duration_last * 1000:.2f} ms",
          f"W-JOB: {self._wkr_3_get_job_dur_tracker.avg_duration_last * 1000:.2f} ms",
          f"W-INFER: {self._wkr_4_inference_dur_tracker.avg_duration_last * 1000:.2f} ms",
          f"W-ADD: {self._wkr_5_add_to_queue_dur_tracker.avg_duration_last * 1000:.2f} ms",
          f"BUSY: {avg_busy_workers:.0f}/{self._n_workers}",
          # f"prel: {avg_preloaded_slots:.0f}",
          # f"busy: {avg_busy_slots:.0f}",
          # f"fill: {avg_filled_slots:.0f}",
        ]

        if self._tot_n_segments_ptr.value > 0:
          progress = (
            self._wkr_total_segments_processed / self._tot_n_segments_ptr.value * 100
          )
          est_remaining_time_s = (
            wall_time
            * (self._tot_n_segments_ptr.value - self._wkr_total_segments_processed)
            / self._wkr_total_segments_processed
          )
          # formatted as HH:MM:SS ohne ms
          est_remaining_time = str(
            datetime.timedelta(seconds=math.ceil(est_remaining_time_s))
          )
          # est_remaining_time = est_remaining_time.split(".")[0]  # remove ms
          output_msg_fields.append(f"PROG: {progress:.1f} %; ETA: {est_remaining_time}")
        else:
          output_msg_fields.append("PROG: analyzing...")

        output_msg = "; ".join(output_msg_fields)
        self._logger.info(output_msg)
        print(output_msg, file=sys.stdout)

        self._next_print = now + self._print_every

    stats = PerformanceTrackingResult(
      worker_speed_xrt=(self._wkr_total_segments_processed * self._segment_size_s)
      / sum(self._wkr_wall_times.values())
      * len(self._wkr_wall_times),
      worker_avg_wall_time_s=(
        sum(self._wkr_wall_times.values()) / len(self._wkr_wall_times)
        if len(self._wkr_wall_times) > 0
        else 0
      ),
      worker_speed_xrt_max=worker_speed_xrt_max,
      total_segments_processed=self._wkr_total_segments_processed,
      total_batches_processed=self._wkr_5_add_to_queue_dur_tracker.n_recordings,
      # summed_prediction_duration_s=self._summed_worker_raw_pred_duration,
      ramp_up_time_until_first_pred_s=self._wkr_ramp_up_time_until_first_pred,
      n_usage_recordings=float_n_records,
      max_memory_usages_MiB=float_max_memory_usage,
      avg_memory_usages_MiB=float_avg_memory_usage,
      max_cpu_usages_pct=float_max_cpu_usage,
      avg_cpu_usages_pct=float_avg_cpu_usage,
      avg_free_slots=float_avg_free_slots,
      avg_busy_slots=float_avg_busy_slots,
      avg_preloaded_slots=float_avg_preloaded_slots,
      avg_busy_workers=float_avg_busy_workers,
      avg_wait_time_ms=self._wkr_1_wait_dur_for_filled_slot_tracker.avg_duration * 1000,
      # avg_pred_dur_last_s=np.mean(self._pred_dur_deque) if self._pred_dur_deque else 0,
      # avg_wait_dur_last_ms=(
      #   np.mean(self._wait_dur_deque) * 1000 if self._wait_dur_deque else 0
      # ),
      # avg_free_slots_last=np.mean(free_slots) if free_slots else 0,
      # avg_busy_slots_last=np.mean(busy_slots) if busy_slots else 0,
      # avg_preloaded_slots_last=np.mean(preloaded_slots) if preloaded_slots else 0,
      # avg_busy_workers_last=np.mean(busy_workers) if busy_workers else 0,
      # max_raw_segments_per_s=max_raw_segments_per_s,
      # avg_segments_per_s_last=(np.mean(avg_segments_per_s) if avg_segments_per_s else 0),
    )

    self._perf_res.put(stats, block=False)

    self._uninit_logging()
