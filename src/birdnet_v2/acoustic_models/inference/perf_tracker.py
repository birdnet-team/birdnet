# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

# You'll need these imports in your own code
import ctypes
import datetime
import math
import multiprocessing
import multiprocessing as mp
import sys
import time
from collections import Counter, deque
from multiprocessing import shared_memory
from multiprocessing.synchronize import Event

# Next two import lines for this demo only
import numpy as np
import psutil
import soundfile as sf  # pip install soundfile

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
import birdnet_v2.logging_utils as bn_logging
from birdnet_v2.globals import READABLE_FLAG, READING_FLAG, WRITABLE_FLAG
from birdnet_v2.helper import (
  RingField,
)


class PerformanceTracker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    pred_dur_queue: mp.SimpleQueue,
    stop_event: mp.Event,
    update_interval: float,
    print_interval: float,
    print_last_n: int,
    start: float,
    logging_queue: mp.Queue,
    logging_level: int,
    perf_res: mp.SimpleQueue,
    chunk_size_s: float,
    parent_process_id: int,
    rf_flags: RingField,
    tot_n_chunks_ptr: ctypes.c_uint64,
    cancel_event: Event = None,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    assert update_interval <= print_interval

    self._perf_res = perf_res
    self._pred_dur_queue = pred_dur_queue
    self._n_last = print_last_n
    self._pred_dur_deque = deque(maxlen=self._n_last)
    self._batch_sizes_deque = deque(maxlen=self._n_last)
    self._update_every = update_interval
    self._stop_event = stop_event
    self._next_print = time.time()
    self._next_update = self._next_print
    self._total_chunks_processed = 0
    self._summed_pred_duration = 0.0
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
    float_max_memory_usage = 0
    float_avg_memory_usage = 0
    float_avg_cpu_usage = 0
    float_max_cpu_usage = 0
    float_avg_free_slots = 0
    float_max_free_slots = 0
    float_avg_filled_slots = 0
    float_avg_busy_slots = 0
    float_avg_preloaded_slots = 0

    cpu_usages = deque(maxlen=self._n_last)
    memory_usages = deque(maxlen=self._n_last)
    free_slots = deque(maxlen=self._n_last)
    filled_slots = deque(maxlen=self._n_last)
    busy_slots = deque(maxlen=self._n_last)
    preloaded_slots = deque(maxlen=self._n_last)

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
        dur, batch_size = self._pred_dur_queue.get()
        self._pred_dur_deque.append(dur)
        self._batch_sizes_deque.append(batch_size)
        self._total_chunks_processed += batch_size
        self._summed_pred_duration += dur
        if ramp_up_time_until_first_pred is None:
          ramp_up_time_until_first_pred = time.perf_counter() - self._start - dur
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

        memory_usage_MiB = memory_usage / (1024 * 1024)
        memory_usages.append(memory_usage_MiB)

        cpu_usage = psutil.cpu_percent()
        cpu_usages.append(cpu_usage)

        c = Counter(self._ring_flags)
        n_free = c.get(WRITABLE_FLAG, 0)
        n_preloaded = c.get(READABLE_FLAG, 0)
        n_busy = c.get(READING_FLAG, 0)
        n_filled = len(self._ring_flags) - n_free
        free_slots.append(n_free)
        filled_slots.append(n_filled)
        busy_slots.append(n_busy)
        preloaded_slots.append(n_preloaded)

        self._next_update = now + self._update_every

      if now >= self._next_print and len(self._pred_dur_deque) > 0:
        perf_duration = time.perf_counter() - self._start
        avg = sum(self._pred_dur_deque) / sum(self._batch_sizes_deque)
        chunks_per_s = self._total_chunks_processed / perf_duration

        memory_usage = parent_process.memory_info().rss
        for child in parent_process.children(recursive=True):
          memory_usage += child.memory_info().rss
        memory_usage_MiB = memory_usage / 1024**2

        cpu_usage = psutil.cpu_percent()

        avg_preloaded_slots = np.mean(preloaded_slots) if preloaded_slots else 0
        avg_free_slots = np.mean(free_slots) if free_slots else 0
        avg_filled_slots = np.mean(filled_slots) if filled_slots else 0
        avg_busy_slots = np.mean(busy_slots) if busy_slots else 0

        output_msg_fields = [
          f"inference speed: {self._summed_pred_duration / self._total_chunks_processed * 1000:.0f} ms/chunk",
          # f"last {len(self._pred_dur_deque)} predictions: {avg * 1000:.0f} ms/chunk",
          f"{chunks_per_s:.0f} chunks/s",
          f"{chunks_per_s * self._chunk_size_s / 60:.2f} min/s",
          f"memory usage: {memory_usage_MiB:.2f} MiB",
          f"CPU usage: {cpu_usage:.1f}%",
          f"prel: {avg_preloaded_slots:.0f}",
          f"free: {avg_free_slots:.0f}",
          f"busy: {avg_busy_slots:.0f}",
          f"fill: {avg_filled_slots:.0f}",
        ]

        if self._tot_n_chunks_ptr.value > 0:
          progress = self._total_chunks_processed / self._tot_n_chunks_ptr.value * 100
          output_msg_fields.append(f"progress: {progress:.2f}%")
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
          output_msg_fields.append(f"remaining: {est_remaining_time}")
        else:
          output_msg_fields.append("progress: analyzing...")

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
    stats["summed_prediction_duration_s"] = self._summed_pred_duration
    stats["ramp_up_time_until_first_pred_s"] = ramp_up_time_until_first_pred
    stats["memory_usages_mb"] = memory_usages
    stats["cpu_usages_pct"] = cpu_usages
    stats["free_slots"] = free_slots
    stats["filled_slots"] = filled_slots
    stats["busy_slots"] = busy_slots
    stats["preloaded_slots"] = preloaded_slots
    self._perf_res.put(stats)

    self._uninit_logging()
