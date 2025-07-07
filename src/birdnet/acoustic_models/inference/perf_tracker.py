# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

# You'll need these imports in your own code
import ctypes
import datetime
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, deque
from multiprocessing import shared_memory
from multiprocessing.synchronize import Event, Semaphore

# Next two import lines for this demo only
import numpy as np
import psutil

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
import birdnet.logging_utils as bn_logging
from birdnet.globals import READABLE_FLAG, READING_FLAG, WRITABLE_FLAG
from birdnet.helper import (
  RingField,
)


class PerformanceTracker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    pred_dur_queue: mp.SimpleQueue,
    stop_event: Event,
    update_interval: float,
    print_interval: float,
    use_stats_from_last_seconds: float,
    n_workers: int,
    start: float,
    workers_start: float,
    logging_queue: mp.Queue,
    logging_level: int,
    perf_res: mp.SimpleQueue,
    sem_active_workers: Semaphore,
    chunk_size_s: float,
    parent_process_id: int,
    rf_flags: RingField,
    tot_n_chunks_ptr: ctypes.c_uint64,
    cancel_event: Event,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    assert update_interval <= print_interval

    self._n_workers = n_workers
    self._workers_start = workers_start
    self._perf_res = perf_res
    self._pred_dur_queue = pred_dur_queue
    self._n_last = int(1 / update_interval * use_stats_from_last_seconds)
    self._sem_active_workers = sem_active_workers
    # self._n_last = print_last_n
    self._wait_dur_deque = deque(maxlen=self._n_last)
    self._pred_dur_deque = deque(maxlen=self._n_last)
    self._batch_sizes_deque = deque(maxlen=self._n_last)
    self._update_every = update_interval
    self._stop_event = stop_event
    self._next_print = time.time()
    self._next_update = self._next_print
    self._total_chunks_processed = 0
    self._total_batches_processed = 0
    self._summed_worker_raw_pred_duration = 0.0
    self._start = start
    self._chunk_size_s = chunk_size_s
    self._parent_process_id = parent_process_id
    self._print_every = print_interval
    self._rf_flags = rf_flags
    self._shm_ring_flags: shared_memory.SharedMemory | None = None
    self._ring_flags: np.ndarray | None = None
    self._tot_n_chunks_ptr = tot_n_chunks_ptr
    self._cancel_event = cancel_event

  def __call__(self):
    self._init_logging()
    self._shm_ring_flags, self._ring_flags = self._rf_flags.attach_and_get_array()
    perf_duration = 0
    ramp_up_time_until_first_pred = None
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
    max_raw_chunks_per_s = 0

    cpu_usages = deque(maxlen=self._n_last)
    memory_usages = deque(maxlen=self._n_last)
    free_slots = deque(maxlen=self._n_last)
    busy_slots = deque(maxlen=self._n_last)
    preloaded_slots = deque(maxlen=self._n_last)
    busy_workers = deque(maxlen=self._n_last)

    summed_warm_up = 0

    avg_chunks_per_s = deque(maxlen=self._n_last)

    worker_wall_time = {}

    cancel = False
    while True:
      if self._cancel_event.is_set():
        cancel = True
        break
      processing_finished = self._stop_event.is_set()
      queue_is_empty = self._pred_dur_queue.empty()
      if processing_finished:
        if queue_is_empty:
          # TODO print again final stats
          break

      while not self._pred_dur_queue.empty():
        worker_pid, warm_up_dur, process_dur, wait_dur, pred_dur, batch_size = (
          self._pred_dur_queue.get()
        )
        worker_wall_time[worker_pid] = process_dur
        self._total_batches_processed += 1
        summed_warm_up += warm_up_dur
        self._wait_dur_deque.append(wait_dur)
        self._pred_dur_deque.append(pred_dur)
        self._batch_sizes_deque.append(batch_size)
        self._total_chunks_processed += batch_size
        self._summed_worker_raw_pred_duration += pred_dur
        if ramp_up_time_until_first_pred is None:
          ramp_up_time_until_first_pred = time.perf_counter() - self._start - pred_dur
          self._logger.info(
            f"Rampup time until first prediction: {ramp_up_time_until_first_pred:.2f}s"
          )

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

        free_slots.append(n_free)
        busy_slots.append(n_busy)
        preloaded_slots.append(n_preloaded)
        busy_workers.append(n_busy_workers)

        float_n_records += 1
        self._next_update = now + self._update_every

      if now >= self._next_print and len(self._pred_dur_deque) > 0:
        t = time.perf_counter()
        perf_duration = t - self._start
        perf_duration_workers = t - self._workers_start
        # avg = sum(self._pred_dur_deque) / sum(self._batch_sizes_deque)
        chunks_per_s = self._total_chunks_processed / perf_duration
        # min_per_s = chunks_per_s * self._chunk_size_s / 60

        memory_usage = parent_process.memory_full_info().uss
        for child in parent_process.children(recursive=True):
          try:
            memory_usage += child.memory_full_info().uss
          except psutil.NoSuchProcess:
            continue
          except psutil.AccessDenied:
            continue
        memory_usage_MiB = memory_usage / 1024**2

        cpu_usage = psutil.cpu_percent()

        avg_preloaded_slots = np.mean(preloaded_slots) if preloaded_slots else 0
        avg_free_slots = np.mean(free_slots) if free_slots else 0
        avg_busy_slots = np.mean(busy_slots) if busy_slots else 0
        avg_busy_workers = np.mean(busy_workers) if busy_workers else 0

        raw_chunks_per_s_old = (
          self._total_chunks_processed
          / (self._summed_worker_raw_pred_duration / avg_busy_workers)
          if avg_busy_workers > 0
          else 0
        )
        processed_audio_duration_s = self._total_chunks_processed * self._chunk_size_s
        real_time_factor_old = (
          (self._summed_worker_raw_pred_duration / avg_busy_workers)
          / processed_audio_duration_s
          if avg_busy_workers > 0
          else 0
        )
        speed_x_real_time_old = (
          processed_audio_duration_s
          / (self._summed_worker_raw_pred_duration / avg_busy_workers)
          if avg_busy_workers > 0
          else 0
        )

        _summed_worker_duration = sum(worker_wall_time.values())

        speed_x_real_time = (
          processed_audio_duration_s / _summed_worker_duration * len(worker_wall_time)
          if _summed_worker_duration > 0
          else 0
        )

        raw_chunks_per_s = (
          self._total_chunks_processed / _summed_worker_duration * len(worker_wall_time)
          if _summed_worker_duration > 0
          else 0
        )

        speed_x_real_time_classic = processed_audio_duration_s / perf_duration

        raw_min_per_s = raw_chunks_per_s_old * self._chunk_size_s / 60

        max_raw_chunks_per_s = max(max_raw_chunks_per_s, raw_chunks_per_s_old)

        avg_chunks_per_s.append(raw_chunks_per_s_old)
        avg_wait_time_ms = (
          np.mean(self._wait_dur_deque) * 1000 if self._wait_dur_deque else 0
        )

        output_msg_fields = [
          # f"inference speed: {self._summed_raw_pred_duration / self._total_chunks_processed * 1000:.0f} ms/chunk",
          # f"last {len(self._pred_dur_deque)} predictions: {avg * 1000:.0f} ms/chunk",
          # f"RTF: {real_time_factor:.8f}x [{raw_chunks_per_s:.0f} segm/s]",
          f"SPEED: {speed_x_real_time:.0f} xRT [{raw_chunks_per_s:.0f} seg/s]",
          f"SPEED2: {speed_x_real_time_classic:.0f} xRT [{chunks_per_s:.0f} seg/s]",
          # f"{raw_min_per_s:.2f} min/s",
          f"MEM: {memory_usage_MiB:.0f} M",
          # f"CPU usage: {cpu_usage:.1f} %",
          f"BUF: {self._ring_flags.shape[0] - avg_free_slots:.0f}/{self._ring_flags.shape[0]}",
          # f"free: {avg_free_slots:.0f}/{self._ring_flags.shape[0]}",
          f"WAIT: {avg_wait_time_ms:.2f} ms",
          f"BUSY: {avg_busy_workers:.0f}/{self._n_workers}",
          # f"prel: {avg_preloaded_slots:.0f}",
          # f"busy: {avg_busy_slots:.0f}",
          # f"fill: {avg_filled_slots:.0f}",
        ]

        if self._tot_n_chunks_ptr.value > 0:
          progress = self._total_chunks_processed / self._tot_n_chunks_ptr.value * 100
          est_remaining_time_s = (
            perf_duration
            * (self._tot_n_chunks_ptr.value - self._total_chunks_processed)
            / self._total_chunks_processed
          )
          # formatted as HH:MM:SS ohne ms
          est_remaining_time = str(
            datetime.timedelta(seconds=math.ceil(est_remaining_time_s))
          )
          # est_remaining_time = est_remaining_time.split(".")[0]  # remove ms
          output_msg_fields.append(f"PROG: {progress:.1f} %; ETA {est_remaining_time}")
        else:
          output_msg_fields.append("PROG: analyzing...")

        output_msg = "; ".join(output_msg_fields)
        self._logger.info(output_msg)
        print(output_msg, file=sys.stdout)

        self._next_print = now + self._print_every

    if cancel:
      self._logger.debug("PerformanceTracker canceled because of cancel event.")
      self._uninit_logging()
      return

    stats = {}
    stats["total_chunks_processed"] = self._total_chunks_processed
    stats["total_batches_processed"] = self._total_batches_processed
    stats["summed_prediction_duration_s"] = self._summed_worker_raw_pred_duration
    stats["ramp_up_time_until_first_pred_s"] = ramp_up_time_until_first_pred
    stats["n_usage_recordings"] = float_n_records

    stats["max_memory_usages_MiB"] = float_max_memory_usage
    stats["avg_memory_usages_MiB"] = float_avg_memory_usage

    stats["max_cpu_usages_pct"] = float_max_cpu_usage
    stats["avg_cpu_usages_pct"] = float_avg_cpu_usage

    stats["avg_free_slots"] = float_avg_free_slots
    stats["avg_busy_slots"] = float_avg_busy_slots
    stats["avg_preloaded_slots"] = float_avg_preloaded_slots
    stats["avg_busy_workers"] = float_avg_busy_workers

    stats["avg_pred_dur_last_s"] = (
      np.mean(self._pred_dur_deque) if self._pred_dur_deque else 0
    )
    stats["avg_wait_dur_last_ms"] = (
      np.mean(self._wait_dur_deque) * 1000 if self._wait_dur_deque else 0
    )
    stats["avg_free_slots_last"] = np.mean(free_slots) if free_slots else 0
    stats["avg_busy_slots_last"] = np.mean(busy_slots) if busy_slots else 0
    stats["avg_preloaded_slots_last"] = (
      np.mean(preloaded_slots) if preloaded_slots else 0
    )
    stats["avg_busy_workers_last"] = np.mean(busy_workers) if busy_workers else 0

    stats["max_raw_chunks_per_s"] = max_raw_chunks_per_s

    stats["avg_chunks_per_s_last"] = (
      np.mean(avg_chunks_per_s) if avg_chunks_per_s else 0
    )

    self._perf_res.put(stats)

    self._uninit_logging()
