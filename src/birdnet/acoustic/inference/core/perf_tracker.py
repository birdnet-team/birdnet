from __future__ import annotations

import ctypes
import datetime
import math
import os
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Queue, shared_memory
from multiprocessing.synchronize import Event
from queue import Empty

import numpy as np
import psutil

import birdnet.acoustic.inference.core.logs as bn_logging
from birdnet.acoustic.inference.core.shm import RingField
from birdnet.acoustic.inference.core.sync import CountedSemaphore, abandon_queue_feeders
from birdnet.globals import READABLE_FLAG, READING_FLAG, WRITABLE_FLAG


@dataclass
class PerformanceTrackingResult:
  worker_speed_xrt: float
  worker_speed_xrt_max: float
  worker_avg_wall_time_s: float
  total_segments_processed: int
  total_batches_processed: int
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


class ValueTracker:
  def __init__(self, n_last: int) -> None:
    self._values = deque(maxlen=n_last)
    self._summed_val = 0
    self._avg_val = 0
    self._min_val = 0
    self._max_val = 0
    self._n_vals = 0

  def add_value(self, val: float | np.floating) -> None:
    val = float(val)
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

  def reset(self) -> None:
    self._values.clear()
    self._summed_val = 0
    self._avg_val = 0
    self._min_val = 0
    self._max_val = 0
    self._n_vals = 0

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
  def vals(self) -> deque[float]:
    return self._values

  @property
  def last_val(self) -> float:
    assert len(self._values) > 0
    return self._values[-1]

  @property
  def median_val(self) -> float:
    if len(self._values) == 0:
      return np.nan
    return float(np.median(self._values))


class PerformanceTracker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    session_id: str,
    pred_dur_queue: Queue,
    prod_stats_queue: Queue,
    callback_queue: Queue | None,
    processing_finished_event: Event,
    update_interval: float,
    n_workers: int,
    logging_queue: Queue,
    logging_level: int,
    perf_res: Queue,
    sem_active_workers: CountedSemaphore,
    sem_filled_slots: CountedSemaphore,
    segment_size_s: float,
    parent_process_id: int,
    rf_flags: RingField,
    tot_n_segments_ptr: ctypes.c_uint64,
    cancel_event: Event,
    end_event: Event,
    start_signal: Event,
    finish_signal: Event,
    start: float,
    start_method: str,
  ) -> None:
    super().__init__(session_id, __name__, logging_queue, logging_level, start_method)

    n_last_batch_stats = 10
    n_last_seconds_live_stats = 5.0
    n_last_live_stats = math.ceil(n_last_seconds_live_stats / update_interval)

    self._sem_filled_slots = sem_filled_slots
    self._processing_finished_event = processing_finished_event
    self._n_workers = n_workers
    self._prd_stats_queue = prod_stats_queue
    self._perf_res = perf_res
    self._wkr_stats_queue = pred_dur_queue
    self._callback_queue = callback_queue

    self._sem_active_workers = sem_active_workers
    self._update_every = update_interval
    self._segment_size_s = segment_size_s

    self._parent_process_id = parent_process_id
    self._parent_process: psutil.Process | None = None
    self._rf_flags = rf_flags
    self._shm_ring_flags: shared_memory.SharedMemory | None = None
    self._ring_flags: np.ndarray | None = None
    self._tot_n_segments_ptr = tot_n_segments_ptr
    self._cancel_event = cancel_event

    self._wkr_wall_times = {}
    self._wkr_total_segments_processed = 0
    self._wkr_1_wait_dur_for_filled_slot_tracker = ValueTracker(n_last_batch_stats)
    self._wkr_2_search_dur_for_filled_slot_tracker = ValueTracker(n_last_batch_stats)
    self._wkr_3_get_job_dur_tracker = ValueTracker(n_last_batch_stats)
    self._wkr_4_copy_to_device_tracker = ValueTracker(n_last_batch_stats)
    self._wkr_5_inference_dur_tracker = ValueTracker(n_last_batch_stats)
    self._wkr_6_add_to_queue_dur_tracker = ValueTracker(n_last_batch_stats)
    self._wkr_busy_tracker = ValueTracker(n_last_live_stats)
    self._wkr_speed_xrt_tracker = ValueTracker(n_last_live_stats)
    self._wkr_speed_seg_per_s_tracker = ValueTracker(n_last_live_stats)

    self._prd_wall_times = {}
    self._prd_total_segments_processed = 0
    self._prd_1_batch_loading_dur_tracker = ValueTracker(n_last_batch_stats)
    self._prd_2_wait_dur_free_slot_tracker = ValueTracker(n_last_batch_stats)
    self._prd_3_free_slot_search_dur_tracker = ValueTracker(n_last_batch_stats)
    self._prd_4_flush_dur_tracker = ValueTracker(n_last_batch_stats)
    self._prd_speed_xrt_tracker = ValueTracker(n_last_live_stats)
    self._prd_speed_seg_per_s_tracker = ValueTracker(n_last_live_stats)

    self._cpu_usage_tracker = ValueTracker(n_last_live_stats)
    self._memory_usage_MiB_tracker = ValueTracker(n_last_live_stats)

    self._rng_free_slots_tracker = ValueTracker(n_last_live_stats)
    self._rng_busy_slots_tracker = ValueTracker(n_last_live_stats)
    self._rng_preloaded_slots_tracker = ValueTracker(n_last_live_stats)

    self._sem_filled_tracker = ValueTracker(n_last_live_stats)
    self._end_event = end_event
    self._start_signal = start_signal
    self._finish_signal = finish_signal
    self._start = start

  def _log(self, message: str) -> None:
    self._logger.debug(f"PT_{os.getpid()}: {message}")

  def _check_cancel_event(self) -> bool:
    if self._cancel_event.is_set():
      self._log("Received cancel event.")
      return True
    return False

  def _check_end_event(self) -> bool:
    if self._end_event.is_set():
      self._log("Received end event.")
      return True
    return False

  def __call__(self) -> None:
    self._init_logging()

    try:
      self._shm_ring_flags, self._ring_flags = self._rf_flags.attach_and_get_array()
      self.run_main_loop()
    except Exception as e:
      self._logger.exception(
        "PerformanceTracker encountered an exception.", exc_info=e, stack_info=True
      )
      self._cancel_event.set()

    # On cancellation the parent stops reading these queues; drop any buffered
    # results/stats so this process can exit without blocking on its feeder
    # threads (see abandon_queue_feeders).
    if self._cancel_event.is_set():
      abandon_queue_feeders(
        self._perf_res,
        self._callback_queue,
        self._wkr_stats_queue,
        self._prd_stats_queue,
      )

    self._uninit_logging()

  def run_main_loop(self) -> None:
    while True:
      self._log("Waiting for start signal...")
      while not self._start_signal.wait(timeout=1.0):
        if self._check_cancel_event():
          # self._uninit_logging()
          return
        if self._check_end_event():
          return

      self._start_signal.clear()
      self._log("Received start signal. Starting processing.")

      self.run_main()

      self._log("Set finish signal.")
      self._finish_signal.set()

  def reset(self) -> None:
    # TODO set to statisticresources start
    self._start = time.perf_counter()

    self._memory_usage_MiB_tracker.reset()
    self._cpu_usage_tracker.reset()
    self._rng_free_slots_tracker.reset()
    self._rng_busy_slots_tracker.reset()
    self._rng_preloaded_slots_tracker.reset()
    self._wkr_busy_tracker.reset()

    self._sem_filled_tracker.reset()

    self._wkr_wall_times.clear()
    self._wkr_total_segments_processed = 0
    self._wkr_1_wait_dur_for_filled_slot_tracker.reset()
    self._wkr_2_search_dur_for_filled_slot_tracker.reset()
    self._wkr_3_get_job_dur_tracker.reset()
    self._wkr_5_inference_dur_tracker.reset()
    self._wkr_6_add_to_queue_dur_tracker.reset()
    self._wkr_speed_xrt_tracker.reset()
    self._wkr_speed_seg_per_s_tracker.reset()

    self._prd_wall_times.clear()
    self._prd_total_segments_processed = 0
    self._prd_1_batch_loading_dur_tracker.reset()
    self._prd_2_wait_dur_free_slot_tracker.reset()
    self._prd_3_free_slot_search_dur_tracker.reset()
    self._prd_4_flush_dur_tracker.reset()
    self._prd_speed_xrt_tracker.reset()
    self._prd_speed_seg_per_s_tracker.reset()

  @staticmethod
  def _safe_proc_memory(proc: psutil.Process) -> float | None:
    try:
      return float(proc.memory_full_info().uss)
    except (psutil.AccessDenied, PermissionError):
      pass
    except psutil.NoSuchProcess:
      return None
    try:
      return float(proc.memory_info().rss)
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
      return None

  def _track_memory_usage(self) -> None:
    if self._parent_process is None:
      try:
        self._parent_process = psutil.Process(self._parent_process_id)
      except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return

    parent_mem = self._safe_proc_memory(self._parent_process)
    if parent_mem is None:
      return

    total = parent_mem
    try:
      children = self._parent_process.children(recursive=True)
    except (psutil.AccessDenied, PermissionError, psutil.NoSuchProcess):
      children = []

    for child in children:
      child_mem = self._safe_proc_memory(child)
      if child_mem is not None:
        total += child_mem

    self._memory_usage_MiB_tracker.add_value(total / 1024**2)

  @property
  def wall_time(self) -> float:
    return time.perf_counter() - self._start

  def run_main(self) -> None:
    self.reset()

    while True:
      if self._check_cancel_event():
        return

      was_empty = self._track_stats()

      if self._processing_finished_event.wait(self._update_every) and was_empty:
        self._log("Processing finished and queues empty.")
        self._callback_stats(finished=True)
        break
      else:
        self._callback_stats(finished=False)

    worker_speed_xrt_max = 0
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
      total_batches_processed=self._wkr_6_add_to_queue_dur_tracker.n_vals,
      # summed_prediction_duration_s=self._summed_worker_raw_pred_duration,
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

    self._log("Putting performance tracking result into queue.")
    self._perf_res.put(stats, block=True)
    # self._perf_res.close()
    # self._perf_res.join_thread()
    self._log("Done putting performance tracking result into queue.")

  def _track_stats(self) -> bool:
    was_empty = self._track_batch_stats()
    self._track_live_stats()
    return was_empty

  def _track_batch_stats(self) -> bool:
    prod_queue_is_empty = self._track_producer_stats()
    worker_queue_is_empty = self._track_worker_stats()
    was_empty = prod_queue_is_empty and worker_queue_is_empty
    return was_empty

  def _track_live_stats(self) -> None:
    self._track_memory_usage()
    self._track_cpu_usage()
    self._track_ring_buffer_stats()
    self._track_semaphore_stats()
    self._track_producer_speed()
    self._track_worker_speed()

  def _track_semaphore_stats(self) -> None:
    self._wkr_busy_tracker.add_value(self._sem_active_workers.get_value())
    # Not strictly the number of occupied ring slots: the shutdown wake-up
    # travels as one extra permit on this semaphore (released by the last
    # producer, relayed by each exiting worker), so this gauge reads one high
    # from producer completion until the permit is consumed or drained.
    self._sem_filled_tracker.add_value(self._sem_filled_slots.get_value())

  def _track_ring_buffer_stats(self) -> None:
    c = Counter(self._ring_flags)
    n_free = c.get(WRITABLE_FLAG, 0)
    n_preloaded = c.get(READABLE_FLAG, 0)
    n_busy = c.get(READING_FLAG, 0)
    self._rng_free_slots_tracker.add_value(n_free)
    self._rng_busy_slots_tracker.add_value(n_busy)
    self._rng_preloaded_slots_tracker.add_value(n_preloaded)

  def _track_cpu_usage(self) -> None:
    cpu_usage = psutil.cpu_percent()
    self._cpu_usage_tracker.add_value(cpu_usage)

  def _track_producer_speed(self) -> None:
    prd_proc_audio_duration_s = (
      self._prd_total_segments_processed * self._segment_size_s
    )
    _summed_prd_duration = sum(self._prd_wall_times.values())
    prd_speed_xrt = float(
      prd_proc_audio_duration_s / _summed_prd_duration * len(self._prd_wall_times)
      if _summed_prd_duration > 0
      else 0
    )
    prd_speed_segments_per_s = float(
      self._prd_total_segments_processed
      / _summed_prd_duration
      * len(self._prd_wall_times)
      if _summed_prd_duration > 0
      else 0
    )
    self._prd_speed_xrt_tracker.add_value(prd_speed_xrt)
    self._prd_speed_seg_per_s_tracker.add_value(prd_speed_segments_per_s)

  def _track_worker_speed(self) -> None:
    wkr_proc_audio_duration_s = (
      self._wkr_total_segments_processed * self._segment_size_s
    )
    _summed_wkr_duration = sum(self._wkr_wall_times.values())
    wkr_speed_segments_per_s = float(
      self._wkr_total_segments_processed
      / _summed_wkr_duration
      * len(self._wkr_wall_times)
      if _summed_wkr_duration > 0
      else 0
    )
    wkr_speed_xrt = float(
      wkr_proc_audio_duration_s / _summed_wkr_duration * len(self._wkr_wall_times)
      if _summed_wkr_duration > 0
      else 0
    )
    self._wkr_speed_xrt_tracker.add_value(wkr_speed_xrt)
    self._wkr_speed_seg_per_s_tracker.add_value(wkr_speed_segments_per_s)

  def _track_worker_stats(self) -> bool:
    # get entries until the marker, entries added in the meanwhile are ignored
    # fix for queue.qsize() which is not supported on macOS
    end_marker = None
    self._wkr_stats_queue.put(end_marker)
    is_empty = True
    n_received = 0
    receive_duration_start = time.perf_counter()

    while True:
      queue_entry = self._wkr_stats_queue.get(block=True)
      if is_end_marker := queue_entry is end_marker:
        break
      is_empty = False
      n_received += 1

      stats: tuple[int, float, float, float, float, float, float, float, int] = (
        queue_entry
      )
      (
        worker_pid,
        wall_time,
        dur_wait_for_filled_slot,
        dur_search_for_filled_slot,
        dur_get_job,
        dur_copy_to_device,
        dur_inference,
        dur_add_to_queue,
        batch_size,
      ) = stats
      # self._log(
      #   f"Received prediction duration from worker {worker_pid}: "
      #   f"wall time: {wall_time:.3f}s, "
      #   f"wait for filled slot: {dur_wait_for_filled_slot:.3f}s, "
      #   f"find filled slot: {dur_search_for_filled_slot:.3f}s, "
      #   f"inference: {dur_inference:.3f}s, add to queue: {dur_add_to_queue}s, "
      #   f"batch size: {batch_size}"
      # )
      self._wkr_wall_times[worker_pid] = wall_time
      self._wkr_total_segments_processed += batch_size

      self._wkr_1_wait_dur_for_filled_slot_tracker.add_value(dur_wait_for_filled_slot)
      self._wkr_2_search_dur_for_filled_slot_tracker.add_value(
        dur_search_for_filled_slot
      )
      self._wkr_3_get_job_dur_tracker.add_value(dur_get_job)
      self._wkr_4_copy_to_device_tracker.add_value(dur_copy_to_device)
      self._wkr_5_inference_dur_tracker.add_value(dur_inference)
      self._wkr_6_add_to_queue_dur_tracker.add_value(dur_add_to_queue)
    self._log(
      f"Received {n_received} worker stats entries in "
      f"{time.perf_counter() - receive_duration_start} ms."
    )
    return is_empty

  def _track_producer_stats(self) -> bool:
    end_marker = None
    self._prd_stats_queue.put(end_marker)
    is_empty = True
    n_received = 0
    receive_duration_start = time.perf_counter()

    while True:
      queue_entry = self._prd_stats_queue.get(block=True)
      if is_end_marker := queue_entry is end_marker:
        break
      is_empty = False
      n_received += 1
      stats: tuple[int, float, float, float, float, float, int] = queue_entry
      (
        prod_pid,
        process_total_duration,
        batch_loading_duration,
        wait_time_for_free_slot,
        free_slot_search_time,
        flush_duration,
        n,
      ) = stats
      # self._log(
      #   f"Received producer stats from producer {prod_pid}: "
      #   f"process: {process_total_duration:.3f}s, "
      #   f"batch loading: {batch_loading_duration:.3f}s, "
      #   f"wait for free slot: {wait_time_for_free_slot:.3f}s, "
      #   f"flush: {flush_duration:.3f}s, n: {n}"
      # )
      self._prd_wall_times[prod_pid] = process_total_duration
      self._prd_total_segments_processed += n

      self._prd_1_batch_loading_dur_tracker.add_value(batch_loading_duration)
      self._prd_2_wait_dur_free_slot_tracker.add_value(wait_time_for_free_slot)
      self._prd_3_free_slot_search_dur_tracker.add_value(free_slot_search_time)
      self._prd_4_flush_dur_tracker.add_value(flush_duration)
    self._log(
      f"Received {n_received} producer stats entries in "
      f"{time.perf_counter() - receive_duration_start} ms."
    )
    return is_empty

  def _callback_stats(self, finished: bool) -> None:
    received_at_least_one_prediction = len(self._wkr_wall_times) > 0
    if not received_at_least_one_prediction:
      return

    p_stats = ProducerStats(
      speed_xrt=self._prd_speed_xrt_tracker.last_val,
      speed_seg_per_s=self._prd_speed_seg_per_s_tracker.last_val,
      wait_ms=self._prd_2_wait_dur_free_slot_tracker.median_val * 1000,
      batch_ms=self._prd_1_batch_loading_dur_tracker.median_val * 1000,
      search_ms=self._prd_3_free_slot_search_dur_tracker.median_val * 1000,
      flush_ms=self._prd_4_flush_dur_tracker.median_val * 1000,
    )

    assert self._ring_flags is not None
    b_stats = BufferStats(
      slots=self._ring_flags.shape[0],
      busy_slots=self._rng_busy_slots_tracker.median_val,
      preloaded_slots=self._rng_preloaded_slots_tracker.median_val,
      free_slots=self._rng_free_slots_tracker.median_val,
    )
    w_stats: WorkerStats | None = None

    processed_batches = 0

    if received_at_least_one_prediction:
      processed_batches = self._wkr_6_add_to_queue_dur_tracker.n_vals

      w_stats = WorkerStats(
        speed_xrt=self._wkr_speed_xrt_tracker.last_val,
        speed_seg_per_s=self._wkr_speed_seg_per_s_tracker.last_val,
        wait_ms=self._wkr_1_wait_dur_for_filled_slot_tracker.median_val * 1000,
        search_ms=self._wkr_2_search_dur_for_filled_slot_tracker.median_val * 1000,
        job_ms=self._wkr_3_get_job_dur_tracker.median_val * 1000,
        copy_ms=self._wkr_4_copy_to_device_tracker.median_val * 1000,
        inference_ms=self._wkr_5_inference_dur_tracker.median_val * 1000,
        add_ms=self._wkr_6_add_to_queue_dur_tracker.median_val * 1000,
        workers=self._n_workers,
        busy=self._wkr_busy_tracker.median_val,
      )

    progress_pct = 0.0
    processed_segments = 0
    total_segments = None
    est_remaining_time_s = None
    wall_time = self.wall_time
    speed_xrt = None
    speed_seg_per_s = None
    if self._tot_n_segments_ptr.value > 0 and self._wkr_total_segments_processed > 0:
      progress_pct = (
        self._wkr_total_segments_processed / self._tot_n_segments_ptr.value * 100
      )
      processed_segments = self._wkr_total_segments_processed
      total_segments = self._tot_n_segments_ptr.value

      est_remaining_time_s = (
        wall_time
        * (self._tot_n_segments_ptr.value - self._wkr_total_segments_processed)
        / self._wkr_total_segments_processed
      )

      speed_xrt = (
        self._wkr_total_segments_processed * self._segment_size_s
      ) / wall_time
      speed_seg_per_s = self._wkr_total_segments_processed / wall_time

    if self._callback_queue is not None:
      stats = AcousticProgressStats(
        finished=finished,
        buffer_stats=b_stats,
        producer_stats=p_stats,
        worker_stats=w_stats,
        wall_time_s=self.wall_time,
        est_remaining_time_s=est_remaining_time_s,
        progress_pct=progress_pct,
        processed_segments=processed_segments,
        total_segments=total_segments,
        processed_batches=processed_batches,
        memory_usage_MiB=self._memory_usage_MiB_tracker.median_val,
        memory_usage_max_MiB=self._memory_usage_MiB_tracker.max_val,
        cpu_usage_pct=self._cpu_usage_tracker.median_val,
        cpu_usage_max_pct=self._cpu_usage_tracker.max_val,
        speed_xrt=speed_xrt,
        speed_seg_per_s=speed_seg_per_s,
      )
      self._callback_queue.put_nowait(stats)


@dataclass
class WorkerStats:
  # x real time
  speed_xrt: float

  # segments per second
  speed_seg_per_s: float

  # median wait time for a filled slot in ms
  wait_ms: float

  # median search time for a filled slot in ms
  search_ms: float

  job_ms: float
  copy_ms: float
  inference_ms: float
  add_ms: float

  # total number of workers
  workers: int

  # median number of busy workers
  busy: float


@dataclass(frozen=True)
class ProducerStats:
  # x real time
  speed_xrt: float

  # segments per second
  speed_seg_per_s: float

  # median wait time for a free slot in buffer in ms
  wait_ms: float

  # median batch loading time in ms
  batch_ms: float

  # median search time for a free slot in ms
  search_ms: float

  # median flush time in ms
  flush_ms: float


@dataclass(frozen=True)
class BufferStats:
  # total number of slots in buffer
  slots: int

  # median number of free slots which can be written to
  free_slots: float

  # median number of slots which are being filled
  busy_slots: float

  # median number of slots which are prepared for reading
  preloaded_slots: float

  # median number of filled slots
  @property
  def filled_slots(self) -> float:
    return self.slots - self.free_slots


@dataclass(frozen=True)
class AcousticProgressStats:
  # processing status
  finished: bool

  # buffer stats
  buffer_stats: BufferStats

  # producer stats
  producer_stats: ProducerStats

  # worker stats, or None if no worker has processed data yet
  worker_stats: WorkerStats | None

  # wall time since start in seconds
  wall_time_s: float

  # memory usage in MiB
  memory_usage_MiB: float

  # maximum memory usage in MiB
  memory_usage_max_MiB: float

  # CPU usage in percentage
  cpu_usage_pct: float

  # total maximum CPU usage in percentage
  cpu_usage_max_pct: float

  # percentage of progress [0-100]
  progress_pct: float

  # estimated remaining time in seconds, or None if still unknown
  est_remaining_time_s: float | None

  @property
  def est_remaining_time_hhmmss(self) -> str | None:
    if self.est_remaining_time_s is None:
      return None
    return str(datetime.timedelta(seconds=math.ceil(self.est_remaining_time_s)))

  # number of processed segments
  processed_segments: int

  # number of processed batches
  processed_batches: int

  # total segments to process, or None if still unknown
  total_segments: int | None

  speed_xrt: float | None

  speed_seg_per_s: float | None


class ProgressDispatcher:
  def __init__(
    self,
    session_id: str,
    callback_queue: Queue,
    callback_fn: Callable[[AcousticProgressStats], None],
    cancel_event: Event,
    end_event: Event,
    start_signal: threading.Event,
    finish_signal: threading.Event,
    processing_finished_event: Event,
    check_interval: float,
  ) -> None:
    self._session_id = session_id
    self._callback_queue = callback_queue
    self._cancel_event = cancel_event
    self._end_event = end_event
    self._start_signal = start_signal
    self._finish_signal = finish_signal
    self._callback_fn = callback_fn
    self._processing_finished_event = processing_finished_event
    self._check_interval = check_interval
    self._logger = bn_logging.get_logger_from_session(session_id, __name__)

  def _log(self, message: str) -> None:
    self._logger.debug(f"PD_{os.getpid()}: {message}")

  def _check_cancel_event(self) -> bool:
    if self._cancel_event.is_set():
      self._log("Received cancel event.")
      return True
    return False

  def _check_end_event(self) -> bool:
    if self._end_event.is_set():
      self._log("Received end event.")
      return True
    return False

  def __call__(self) -> None:
    try:
      self.run_main_loop()
    except Exception as e:
      self._logger.exception(
        "ProgressDispatcher encountered an exception.", exc_info=e, stack_info=True
      )
      self._cancel_event.set()

  def run_main_loop(self) -> None:
    while True:
      while not self._start_signal.wait(timeout=1.0):
        if self._check_cancel_event():
          # self._uninit_logging()
          return
        if self._check_end_event():
          return

      self._start_signal.clear()
      self._log("Received start signal. Starting processing.")
      self.run_main()

      self._log("Set finish signal.")
      self._finish_signal.set()

  def get_last_stats(self) -> AcousticProgressStats | None:
    latest = None

    while True:
      try:
        latest = self._callback_queue.get(block=True, timeout=self._check_interval)
      except Empty:
        break

    return latest

  def run_main(self) -> None:
    while True:
      latest = self.get_last_stats()

      if latest is None:
        # it has started, so ending is not possible, only canceling
        if self._check_cancel_event():
          return
      else:
        self._log("Received stats. Call callback function.")
        self._callback_fn(latest)

        finished = self._processing_finished_event.is_set()
        if finished and latest.finished:
          # last stats should always be called
          self._log("Processing finished event is set. Exiting loop.")
          return
