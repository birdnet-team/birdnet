from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
import os
import time
from abc import abstractmethod
from multiprocessing import Queue, shared_memory
from multiprocessing.synchronize import Event, Semaphore

import numpy as np
from numpy.typing import DTypeLike

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.backends import InferenceBackendLoader
from birdnet.globals import (
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
)
from birdnet.helper import RingField


class WorkerBase(bn_logging.LogableProcessBase):
  def __init__(
    self,
    name: str,
    backend_loader: InferenceBackendLoader,
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_segment_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    segment_duration_samples: int,
    out_q: Queue,
    wkr_ring_access_lock: multiprocessing.synchronize.Lock,
    sem_free: Semaphore,
    sem_fill: Semaphore,
    sem_active_workers: Semaphore | None,
    infer_dtype: DTypeLike,
    wkr_stats_queue: mp.Queue | None,
    logging_queue: mp.Queue,
    logging_level: int,
    device: str,
    cancel_event: Event,
    prd_all_done_event: Event,
  ):
    super().__init__(name, logging_queue, logging_level)

    self._backend_loader = backend_loader
    self._prd_all_done_event = prd_all_done_event
    self._wkr_ring_access_lock = wkr_ring_access_lock
    self._backend = None
    self._wkr_stats_queue = wkr_stats_queue
    self._out_q = out_q
    self._sem_free = sem_free
    self._sem_filled = sem_fill
    self._sem_active_workers = sem_active_workers
    self._prediction_count = 0
    assert np.dtype(infer_dtype) in (np.float16, np.float32)
    self._infer_dtype = infer_dtype
    # Interpreter
    self._slot = 0
    self._batch_idx_cache = {}

    self._n_slots = n_slots
    self._batch_size = batch_size
    self._segment_duration_samples = segment_duration_samples

    self._species_dtype: DTypeLike | None = None

    self._rf_file_indices = rf_file_indices
    self._rf_segment_indices = rf_segment_indices
    self._rf_audio_samples = rf_audio_samples
    self._rf_batch_sizes = rf_batch_sizes
    self._rf_flags = rf_flags

    self._in_idx: int | None = None
    self._out_idx: int | None = None

    self._shm_file_indices: shared_memory.SharedMemory | None = None
    self._shm_segment_indices: shared_memory.SharedMemory | None = None
    self._shm_audio_samples: shared_memory.SharedMemory | None = None
    self._shm_batch_sizes: shared_memory.SharedMemory | None = None
    self._shm_ring_flags: shared_memory.SharedMemory | None = None

    self._ring_file_indices: np.ndarray | None = None
    self._ring_segment_indices: np.ndarray | None = None
    self._ring_audio_samples: np.ndarray | None = None
    self._ring_batch_sizes: np.ndarray | None = None
    self._ring_flags: np.ndarray | None = None
    self._device_name = device
    # self._mm: mmap.mmap | None = None

    self._cancel_event = cancel_event

    self._lazy_init = mp.get_start_method() != "fork"

    if not self._lazy_init:
      self._init_logging()
      self._load_ring_buffers()

  def _load_model(self) -> None:
    self._log_debug("Loading model...")
    try:
      self._backend = self._backend_loader.load_backend()
    except ValueError as e:
      self._log_debug(f"Failed to load model: {e}")
      raise e
    self._log_debug("Model loaded.")

  def _infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._backend is not None
    res = self._backend.infer(batch, self._device_name)
    assert res.dtype == np.float32
    res = res.astype(self._infer_dtype, copy=False)
    return res

  def _load_ring_buffers(self) -> None:
    self._log_debug("Attaching ring buffers...")
    # attach to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not delete the shared memory access
    self._shm_file_indices, self._ring_file_indices = (
      self._rf_file_indices.attach_and_get_array()
    )
    self._shm_segment_indices, self._ring_segment_indices = (
      self._rf_segment_indices.attach_and_get_array()
    )
    self._shm_audio_samples, self._ring_audio_samples = (
      self._rf_audio_samples.attach_and_get_array()
    )
    self._shm_batch_sizes, self._ring_batch_sizes = (
      self._rf_batch_sizes.attach_and_get_array()
    )
    self._shm_ring_flags, self._ring_flags = self._rf_flags.attach_and_get_array()
    self._log_debug("Attached ring buffers.")

  def _uninit(self) -> None:
    self._uninit_logging()

  @property
  def _pid(self) -> int:
    return os.getpid()

  def _log_debug(self, msg: str) -> None:
    self._logger.debug(f"WORKER({self._pid}) - {msg}")

  def _check_cancel_event(self) -> bool:
    if self._cancel_event.is_set():
      self._log_debug("Received cancel event.")
      return True
    return False

  @abstractmethod
  def _get_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    infer_result: np.ndarray,
  ) -> tuple[np.ndarray, ...]: ...

  def __call__(self):
    start = time.perf_counter()

    if self._lazy_init:
      self._init_logging()
      self._load_ring_buffers()

    try:
      self._load_model()
    except ValueError:
      self._cancel_event.set()
      return
    duration_init = time.perf_counter() - start
    self._log_debug(f"Worker {self._pid} initialized in {duration_init:.4f} seconds.")

    assert self._ring_flags is not None
    assert self._ring_file_indices is not None
    assert self._ring_segment_indices is not None
    assert self._ring_audio_samples is not None
    assert self._ring_batch_sizes is not None

    start_time = time.perf_counter()

    while True:
      perf_c = time.perf_counter()
      while not self._sem_filled.acquire(timeout=1.0):
        if self._check_cancel_event():
          return
        if self._prd_all_done_event.is_set():
          self._log_debug("Producer is done. Exiting worker.")
          self._out_q.put(None)
          return
      dur_wait_for_filled_slot = time.perf_counter() - perf_c

      self._log_debug(
        f"Acquired FILL; Free slots remaining: {self._sem_free}; Filled slots: {self._sem_filled}"
      )

      if self._check_cancel_event():
        return

      claimed_slot = None
      claimed_flag = None

      perf_c = time.perf_counter()
      with self._wkr_ring_access_lock:
        for current_slot in range(self._n_slots):
          current_slot_flag = self._ring_flags[current_slot]

          # TODO: check if all ring_size slots = DONE
          if current_slot_flag == READABLE_FLAG:
            claimed_slot = current_slot
            claimed_flag = current_slot_flag
            self._ring_flags[claimed_slot] = READING_FLAG
            break
          else:
            assert current_slot_flag in (
              WRITABLE_FLAG,
              WRITING_FLAG,
              READING_FLAG,
            )

      dur_search_for_filled_slot = time.perf_counter() - perf_c

      if claimed_slot is None:
        if self._prd_all_done_event.is_set():
          self._log_debug("Producer is done. Exiting worker.")
          self._out_q.put(None)
          break
        # if n_done >= 1:
        #   self._log_debug(
        #     f"Slots are DONE_FLAG {claimed_slot}. No more work to d. Exiting."
        #   )
        #   self._out_q.put(None)
        #   break
        else:
          raise AssertionError(
            "No slot found in the ring buffer but sem_fill was available!"
          )

      assert claimed_flag == READABLE_FLAG

      self._log_debug(
        f"Acquired READ_FLAG for slot {claimed_slot}. Searched {dur_search_for_filled_slot:.4f} seconds for batch."
      )

      if self._sem_active_workers is not None:
        self._sem_active_workers.release()

      perf_c = time.perf_counter()
      n = self._ring_batch_sizes[claimed_slot]
      audio_samples = self._ring_audio_samples[claimed_slot, :n]
      file_indices = self._ring_file_indices[claimed_slot, :n].copy()  # copy needed
      segment_indices = self._ring_segment_indices[
        claimed_slot, :n
      ].copy()  # copy needed
      dur_get_job = time.perf_counter() - perf_c
      self._log_debug(
        f"Received job for slot {claimed_slot} with {n} segments: {segment_indices}"
      )

      perf_c = time.perf_counter()
      try:
        infer_result = self._infer(audio_samples)
      except Exception as e:
        self._log_debug(f"Error during inference: {e}")
        self._cancel_event.set()
        self._log_debug(
          f"Exiting worker {self._pid} due to error during inference. Set cancel event."
        )
        return
      dur_inference = time.perf_counter() - perf_c

      self._ring_flags[claimed_slot] = WRITABLE_FLAG
      self._sem_free.release()

      self._log_debug(
        f"Released FREE. Free slots remaining: {self._sem_free}; Filled slots: {self._sem_filled}"
      )

      assert infer_result.flags.aligned

      perf_c = time.perf_counter()
      block = self._get_block(file_indices, segment_indices, infer_result)
      self._out_q.put(block)
      dur_add_to_queue = time.perf_counter() - perf_c

      self._prediction_count += n
      self._log_debug(
        f"Prediction made ({dur_inference:.4} s). Total predictions: {self._prediction_count}. Chunks: {segment_indices}"
      )

      if self._wkr_stats_queue is not None:
        wall_time = time.perf_counter() - start_time
        self._wkr_stats_queue.put(
          (
            self._pid,
            wall_time,
            dur_wait_for_filled_slot,
            dur_search_for_filled_slot,
            dur_get_job,
            dur_inference,
            dur_add_to_queue,
            n,
          ),
          block=False,
        )

      if self._sem_active_workers is not None:
        self._sem_active_workers.acquire(block=False)

    self._log_debug("Finished.")
    self._uninit()
