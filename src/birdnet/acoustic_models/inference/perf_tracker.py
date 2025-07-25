from __future__ import annotations

import ctypes
import datetime
import math
import multiprocessing as mp
import sys
import threading as th
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


class ValueTracker:
  def __init__(self, n_last: int):
    self._values = deque(maxlen=n_last)
    self._summed_val = 0
    self._avg_val = 0
    self._min_val = 0
    self._max_val = 0
    self._n_vals = 0

  def add_value(self, val: float) -> None:
    self._values.append(val)
    self._min_val = min(self._min_val, val) if self._n_vals > 0 else val
    self._max_val = max(self._max_val, val) if self._n_vals > 0 else val
    self._summed_val = self._summed_val + val if self._n_vals > 0 else val
    self._avg_val = (
      (self._avg_val * self._n_vals + val) / (self._n_vals + 1)
      if self._n_vals > 0
      else val
    )

    self._n_vals += 1

  @property
  def avg_val(self) -> float:
    return self._avg_val

  @property
  def min_val(self) -> float:
    return self._min_val

  @property
  def max_val(self) -> float:
    return self._max_val

  @property
  def summed_val(self) -> float:
    return self._summed_val

  @property
  def n_vals(self) -> int:
    return self._n_vals

  @property
  def vals_last(self) -> deque[float]:
    return self._values

  @property
  def avg_val_last(self) -> float:
    if len(self._values) == 0:
      return np.nan
    return np.mean(self._values)  # type: ignore


class PerformanceTracker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    pred_dur_queue: mp.Queue,
    prod_stats_queue: mp.Queue,
    processing_finished_event: Event,
    update_interval: float,
    print_interval: float,
    n_workers: int,
    start: float,
    workers_start: float,
    logging_queue: mp.Queue,
    logging_level: int,
    perf_res: mp.Queue,
    sem_active_workers: Semaphore,
    sem_filled_slots: Semaphore,
    segment_size_s: float,
    parent_process_id: int,
    rf_flags: RingField,
    tot_n_segments_ptr: ctypes.c_uint64,
    cancel_event: Event,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    assert update_interval <= print_interval

    update_interval = 1.0
    print_interval = 2.0
    n_last_batches = 20
    n_last_updated_s = 5.0
    n_last_updated = math.ceil(n_last_updated_s / update_interval)

    self._sem_filled_slots = sem_filled_slots
    self._processing_finished_event = processing_finished_event
    self._n_workers = n_workers
    self._workers_start = workers_start
    self._prd_stats_queue = prod_stats_queue
    self._perf_res = perf_res
    self._wkr_stats_queue = pred_dur_queue

    self._sem_active_workers = sem_active_workers
    self._update_every = update_interval
    self._start = start
    self._segment_size_s = segment_size_s

    self._parent_process_id = parent_process_id
    self._parent_process: psutil.Process | None = None
    self._print_interval = print_interval
    self._rf_flags = rf_flags
    self._shm_ring_flags: shared_memory.SharedMemory | None = None
    self._ring_flags: np.ndarray | None = None
    self._tot_n_segments_ptr = tot_n_segments_ptr
    self._cancel_event = cancel_event

    self._wkr_wall_times = {}
    self._wkr_total_segments_processed = 0
    self._wkr_1_wait_dur_for_filled_slot_tracker = ValueTracker(n_last_batches)
    self._wkr_2_search_dur_for_filled_slot_tracker = ValueTracker(n_last_batches)
    self._wkr_3_get_job_dur_tracker = ValueTracker(n_last_batches)
    self._wkr_4_inference_dur_tracker = ValueTracker(n_last_batches)
    self._wkr_5_add_to_queue_dur_tracker = ValueTracker(n_last_batches)
    self._wkr_ramp_up_time_until_first_pred = None
    self._wkr_busy_tracker = ValueTracker(n_last_updated)

    self._prd_wall_times = {}
    self._prd_total_segments_processed = 0
    self._prd_1_batch_loading_dur_tracker = ValueTracker(n_last_batches)
    self._prd_2_wait_dur_free_slot_tracker = ValueTracker(n_last_batches)
    self._prd_3_free_slot_search_dur_tracker = ValueTracker(n_last_batches)
    self._prd_4_flush_dur_tracker = ValueTracker(n_last_batches)

    self._cpu_usage_tracker = ValueTracker(n_last_updated)
    self._memory_usage_MiB_tracker = ValueTracker(n_last_updated)

    self._rng_free_slots_tracker = ValueTracker(n_last_updated)
    self._rng_busy_slots_tracker = ValueTracker(n_last_updated)
    self._rng_preloaded_slots_tracker = ValueTracker(n_last_updated)

    self._sem_filled_tracker = ValueTracker(n_last_updated)

  def _get_worker_stats(self) -> None:
    entry_count = self._wkr_stats_queue.qsize()
    for _ in range(entry_count):
      (
        worker_pid,
        wall_time,
        dur_wait_for_filled_slot,
        dur_search_for_filled_slot,
        dur_get_job,
        dur_inference,
        dur_add_to_queue,
        batch_size,
      ) = self._wkr_stats_queue.get(block=True)
      self._logger.debug(
        f"PerformanceTracker received prediction duration from worker {worker_pid}: "
        f"wall time: {wall_time:.3f}s, wait for filled slot: {dur_wait_for_filled_slot:.3f}s, find filled slot: {dur_search_for_filled_slot:.3f}s, inference: {dur_inference:.3f}s, add to queue: {dur_add_to_queue}s, batch size: {batch_size}"
      )
      self._wkr_wall_times[worker_pid] = wall_time
      self._wkr_total_segments_processed += batch_size

      self._wkr_1_wait_dur_for_filled_slot_tracker.add_value(dur_wait_for_filled_slot)
      self._wkr_2_search_dur_for_filled_slot_tracker.add_value(
        dur_search_for_filled_slot
      )
      self._wkr_3_get_job_dur_tracker.add_value(dur_get_job)
      self._wkr_4_inference_dur_tracker.add_value(dur_inference)
      self._wkr_5_add_to_queue_dur_tracker.add_value(dur_add_to_queue)

      if self._wkr_ramp_up_time_until_first_pred is None:
        self._wkr_ramp_up_time_until_first_pred = (
          time.perf_counter() - self._start - dur_inference
        )
        self._logger.info(
          f"Rampup time until first prediction: {self._wkr_ramp_up_time_until_first_pred:.2f}s"
        )

  def _get_producer_stats(self) -> None:
    entry_count = self._prd_stats_queue.qsize()
    for _ in range(entry_count):
      (
        prod_pid,
        process_total_duration,
        batch_loading_duration,
        wait_time_for_free_slot,
        free_slot_search_time,
        flush_duration,
        n,
      ) = self._prd_stats_queue.get(block=True)
      self._logger.debug(
        f"PerformanceTracker received producer stats from producer {prod_pid}: "
        f"process: {process_total_duration:.3f}s, batch loading: {batch_loading_duration:.3f}s, "
        f"wait for free slot: {wait_time_for_free_slot:.3f}s, flush: {flush_duration:.3f}s, n: {n}"
      )
      self._prd_wall_times[prod_pid] = process_total_duration
      self._prd_total_segments_processed += n

      self._prd_1_batch_loading_dur_tracker.add_value(batch_loading_duration)
      self._prd_2_wait_dur_free_slot_tracker.add_value(wait_time_for_free_slot)
      self._prd_3_free_slot_search_dur_tracker.add_value(free_slot_search_time)
      self._prd_4_flush_dur_tracker.add_value(flush_duration)

  def _print_stats(self) -> None:
    received_at_least_one_prediction = len(self._wkr_wall_times) > 0
    if not received_at_least_one_prediction:
      return

    wall_time = time.perf_counter() - self._start
    # perf_duration_workers = t - self._workers_start
    # avg = sum(self._pred_dur_deque) / sum(self._batch_sizes_deque)
    # segments_per_s = self._total_segments_processed / wall_time
    # min_per_s = segments_per_s * self._segment_size_s / 60
    if self._parent_process is None:
      self._parent_process = psutil.Process(self._parent_process_id)
    memory_usage = self._parent_process.memory_full_info().uss
    for child in self._parent_process.children(recursive=True):
      try:
        memory_usage += child.memory_full_info().uss
      except psutil.NoSuchProcess:
        continue
      except psutil.AccessDenied:
        continue
    memory_usage_MiB = memory_usage / 1024**2

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
    assert self._ring_flags is not None

    output_msg_fields = [
      # f"inference speed: {self._summed_raw_pred_duration / self._total_segments_processed * 1000:.0f} ms/segment",
      # f"last {len(self._pred_dur_deque)} predictions: {avg * 1000:.0f} ms/segment",
      # f"RTF: {real_time_factor:.8f}x [{raw_segments_per_s:.0f} segm/s]",
      # f"SPEED2: {speed_x_real_time_classic:.0f} xRT [{segments_per_s:.0f} seg/s]",
      # f"{raw_min_per_s:.2f} min/s",
      f"MEM: {memory_usage_MiB:.0f} M",
      # f"CPU usage: {cpu_usage:.1f} %",
      f"BUF: {self._ring_flags.shape[0] - self._rng_free_slots_tracker.avg_val_last:.0f}/{self._ring_flags.shape[0]}",
      # f"BUF2: {self._rng_preloaded_slots_tracker.avg_val_last:.0f}/{self._ring_flags.shape[0]}",
      # f"S-FILL: {self._sem_filled_tracker.avg_val_last:.0f}",
      # f"free: {avg_free_slots:.0f}/{self._ring_flags.shape[0]}",
      f"F-SPEED: {prd_speed_xrt:.0f} xRT [{prd_speed_segments_per_s:.0f} seg/s]",
      f"F-WAIT: {self._prd_1_batch_loading_dur_tracker.avg_val_last * 1000:.2f} ms",
      f"F-BATCH: {self._prd_2_wait_dur_free_slot_tracker.avg_val_last * 1000:.2f} ms",
      f"F-SEARCH: {self._prd_3_free_slot_search_dur_tracker.avg_val_last * 1000:.2f} ms",
      f"F-FLUSH: {self._prd_4_flush_dur_tracker.avg_val_last * 1000:.2f} ms",
    ]
    if received_at_least_one_prediction:
      output_msg_fields += [
        f"W-SPEED: {wkr_speed_xrt:.0f} xRT [{wkr_speed_segments_per_s:.0f} seg/s]",
        f"W-WAIT: {self._wkr_1_wait_dur_for_filled_slot_tracker.avg_val * 1000:.2f} ms",
        f"W-SEARCH: {self._wkr_2_search_dur_for_filled_slot_tracker.avg_val_last * 1000:.2f} ms",
        f"W-JOB: {self._wkr_3_get_job_dur_tracker.avg_val_last * 1000:.2f} ms",
        f"W-INFER: {self._wkr_4_inference_dur_tracker.avg_val_last * 1000:.2f} ms",
        f"W-ADD: {self._wkr_5_add_to_queue_dur_tracker.avg_val_last * 1000:.2f} ms",
        f"BUSY: {self._wkr_busy_tracker.avg_val_last:.0f}/{self._n_workers}",
        # f"prel: {avg_preloaded_slots:.0f}",
        # f"busy: {avg_busy_slots:.0f}",
        # f"fill: {avg_filled_slots:.0f}",
      ]
    else:
      output_msg_fields += [
        "W: loading model...",
      ]

    if self._tot_n_segments_ptr.value > 0 and self._wkr_total_segments_processed > 0:
      progress = (
        self._wkr_total_segments_processed / self._tot_n_segments_ptr.value * 100
      )
      output_msg_fields.append(f"PROG: {progress:.1f} %")

      est_remaining_time_s = (
        wall_time
        * (self._tot_n_segments_ptr.value - self._wkr_total_segments_processed)
        / self._wkr_total_segments_processed
      )
      # formatted as HH:MM:SS ohne ms
      est_remaining_time = str(
        datetime.timedelta(seconds=math.ceil(est_remaining_time_s))
      )
      output_msg_fields.append(f"ETA: {est_remaining_time}")
    else:
      output_msg_fields.append("PROG: analyzing...")

    output_msg = "; ".join(output_msg_fields)
    # self._logger.info(output_msg)
    print(output_msg, file=sys.stdout)

  def print_stats_continuously(self):
    while not self._processing_finished_event.wait(self._print_interval):
      if self._cancel_event.is_set():
        return
      self._print_stats()
    self._logger.debug("PerformanceTracker.PrintThread thread finished.")

  def __call__(self):
    self._init_logging()
    self._shm_ring_flags, self._ring_flags = self._rf_flags.attach_and_get_array()

    print_thread = th.Thread(
      target=self.print_stats_continuously,
      name="PerformanceTracker.PrintThread",
      daemon=True,
    )
    print_thread.start()

    worker_speed_xrt_max = 0

    while (
      not self._processing_finished_event.is_set()
      or self._wkr_stats_queue.qsize() != 0
      or self._prd_stats_queue.qsize() != 0
    ):
      if self._cancel_event.is_set():
        self._logger.debug("PerformanceTracker canceled because of cancel event.")
        return

      self._get_producer_stats()
      self._get_worker_stats()

      if not self._processing_finished_event.wait(self._update_every):
        if self._parent_process is None:
          self._parent_process = psutil.Process(self._parent_process_id)
        memory_usage: float = self._parent_process.memory_full_info().uss
        for child in self._parent_process.children(recursive=True):
          try:
            memory_usage += child.memory_full_info().uss
          except psutil.NoSuchProcess:
            continue
          except psutil.AccessDenied:
            continue

        self._memory_usage_MiB_tracker.add_value(memory_usage / 1024**2)

        cpu_usage = psutil.cpu_percent()
        self._cpu_usage_tracker.add_value(cpu_usage)

        c = Counter(self._ring_flags)
        n_free = c.get(WRITABLE_FLAG, 0)
        n_preloaded = c.get(READABLE_FLAG, 0)
        n_busy = c.get(READING_FLAG, 0)

        self._rng_free_slots_tracker.add_value(n_free)
        self._rng_busy_slots_tracker.add_value(n_busy)
        self._rng_preloaded_slots_tracker.add_value(n_preloaded)
        self._wkr_busy_tracker.add_value(self._sem_active_workers.get_value())

        self._sem_filled_tracker.add_value(self._sem_filled_slots.get_value())

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

    stats = PerformanceTrackingResult(
      worker_speed_xrt=(self._wkr_total_segments_processed * self._segment_size_s)
      / sum(self._wkr_wall_times.values())
      * len(self._wkr_wall_times)
      if len(self._wkr_wall_times) > 0
      else 0,
      worker_avg_wall_time_s=(
        sum(self._wkr_wall_times.values()) / len(self._wkr_wall_times)
        if len(self._wkr_wall_times) > 0
        else 0
      ),
      worker_speed_xrt_max=worker_speed_xrt_max,
      total_segments_processed=self._wkr_total_segments_processed,
      total_batches_processed=self._wkr_5_add_to_queue_dur_tracker.n_vals,
      # summed_prediction_duration_s=self._summed_worker_raw_pred_duration,
      ramp_up_time_until_first_pred_s=self._wkr_ramp_up_time_until_first_pred,
      n_usage_recordings=self._memory_usage_MiB_tracker.n_vals,
      max_memory_usages_MiB=self._memory_usage_MiB_tracker.max_val,
      avg_memory_usages_MiB=self._memory_usage_MiB_tracker.avg_val,
      max_cpu_usages_pct=self._cpu_usage_tracker.max_val,
      avg_cpu_usages_pct=self._cpu_usage_tracker.avg_val,
      avg_free_slots=self._rng_free_slots_tracker.avg_val,
      avg_busy_slots=self._rng_busy_slots_tracker.avg_val,
      avg_preloaded_slots=self._rng_preloaded_slots_tracker.avg_val,
      avg_busy_workers=self._wkr_busy_tracker.avg_val,
      avg_wait_time_ms=self._wkr_1_wait_dur_for_filled_slot_tracker.avg_val * 1000,
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

    self._logger.info("Joining print thread...")
    print_thread.join()
    self._logger.info("Print thread joined.")

    self._logger.debug("Putting performance tracking result into queue.")
    self._perf_res.put(stats, block=True)
    # self._perf_res.close()
    # self._perf_res.join_thread()
    self._logger.debug("Done putting performance tracking result into queue.")

    self._uninit_logging()
