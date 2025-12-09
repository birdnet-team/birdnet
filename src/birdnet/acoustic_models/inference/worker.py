from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
import os
import time
from abc import abstractmethod
from multiprocessing import Queue, shared_memory
from multiprocessing.synchronize import Event, Semaphore
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

import birdnet.acoustic_models.inference_pipeline.logs as bn_logging
from birdnet.backends import BackendLoader, BatchT, VersionedBackendProtocol
from birdnet.globals import (
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
)
from birdnet.shm import RingField

if TYPE_CHECKING:
  pass


class WorkerBase(bn_logging.LogableProcessBase):
  def __init__(
    self,
    session_id: str,
    name: str,
    backend_loader: BackendLoader,
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
    half_precision: bool,
    wkr_stats_queue: Queue | None,
    logging_queue: Queue,
    logging_level: int,
    device: str,
    cancel_event: Event,
    all_producers_finished: Event,
    start_signal: Event,
    end_event: Event,
  ) -> None:
    super().__init__(session_id, name, logging_queue, logging_level)

    self._half_precision = half_precision
    self._end_event = end_event
    self._start_signal = start_signal
    self._backend_loader = backend_loader
    self._all_producers_finished = all_producers_finished
    self._wkr_ring_access_lock = wkr_ring_access_lock
    self._backend: VersionedBackendProtocol | None = None
    self._wkr_stats_queue = wkr_stats_queue
    self._out_q = out_q
    self._sem_free = sem_free
    self._sem_filled = sem_fill
    self._sem_active_workers = sem_active_workers
    self._prediction_count = 0
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
    self._log("Loading model...")
    try:
      self._backend = self._backend_loader.load_backend(
        self._device_name, half_precision=self._half_precision
      )
    except ValueError as e:
      self._log(f"Failed to load model: {e}")
      raise e
    self._log("Model loaded.")

  @abstractmethod
  def _infer(self, batch: BatchT) -> BatchT: ...

  def _load_ring_buffers(self) -> None:
    self._log("Attaching ring buffers...")
    # attach to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not
    # delete the shared memory access
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
    self._log("Attached ring buffers.")

  def _uninit(self) -> None:
    self._uninit_logging()

  @property
  def _pid(self) -> int:
    return os.getpid()

  def _log(self, msg: str) -> None:
    self._logger.debug(f"W_{self._pid}: {msg}")

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

  @abstractmethod
  def _get_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    infer_result: np.ndarray,
  ) -> tuple[np.ndarray, ...]: ...

  def __call__(self) -> None:
    start = time.perf_counter()

    if self._lazy_init:
      self._init_logging()

    try:
      if self._lazy_init:
        self._load_ring_buffers()

      self._load_model()

      duration_init = time.perf_counter() - start
      self._log(f"Initialized in {duration_init:.4f} seconds.")

      self.run_main_loop()

      self._log("Finished.")
    except Exception as e:
      self._logger.exception(
        "Worker encountered an exception.", exc_info=e, stack_info=True
      )
      self._cancel_event.set()

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

  def run_main(self) -> None:
    assert self._backend is not None
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

        if self._all_producers_finished.is_set():
          self._log("Producer is done. Exiting worker.")
          self._out_q.put(None)
          return
      dur1_wait_for_filled_slot = time.perf_counter() - perf_c

      self._log(
        f"Acquired FILL; Free slots remaining: {self._sem_free}; "
        f"Filled slots: {self._sem_filled}"
      )

      if self._check_cancel_event():
        return

      claimed_slot = None
      claimed_flag = None

      perf_c = time.perf_counter()
      dur2_search_for_filled_slot = None
      with self._wkr_ring_access_lock:
        for current_slot in range(self._n_slots):
          current_slot_flag = self._ring_flags[current_slot]

          # TODO: check if all ring_size slots = DONE
          if current_slot_flag == READABLE_FLAG:
            dur2_search_for_filled_slot = time.perf_counter() - perf_c
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

      assert dur2_search_for_filled_slot is not None

      if claimed_slot is None:
        if self._all_producers_finished.is_set():
          self._log("Producer is done. Exiting worker.")
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

      self._log(
        f"Acquired READ_FLAG for slot {claimed_slot}. "
        f"Searched {dur2_search_for_filled_slot:.4f} seconds for batch."
      )

      if self._sem_active_workers is not None:
        self._sem_active_workers.release()

      perf_c = time.perf_counter()
      n = int(self._ring_batch_sizes[claimed_slot])
      audio_samples = self._ring_audio_samples[claimed_slot, :n]
      file_indices = self._ring_file_indices[claimed_slot, :n].copy()  # copy needed
      segment_indices = self._ring_segment_indices[
        claimed_slot, :n
      ].copy()  # copy needed
      dur3_get_job = time.perf_counter() - perf_c
      self._log(
        f"Received job for slot {claimed_slot} with {n} segments: {segment_indices}"
      )

      perf_c = time.perf_counter()
      tensor = self._backend.copy_to_device(audio_samples)
      dur4_copy_to_device = time.perf_counter() - perf_c
      perf_c = time.perf_counter()

      try:
        infer_result_tensor = self._infer(tensor)
      except Exception as e:
        print(e)
        self._log(f"Error during inference: {e}")
        self._cancel_event.set()
        self._log("Exiting due to error during inference. Set cancel event.")
        return
      dur5_inference = time.perf_counter() - perf_c

      self._ring_flags[claimed_slot] = WRITABLE_FLAG
      self._sem_free.release()

      self._log(
        f"Released FREE. Free slots remaining: {self._sem_free}; "
        f"Filled slots: {self._sem_filled}"
      )

      perf_c = time.perf_counter()
      if self._half_precision:
        infer_result_tensor = self._backend.half_precision(infer_result_tensor)
      dur_half_precision = time.perf_counter() - perf_c

      perf_c = time.perf_counter()
      infer_result = self._backend.copy_from_device(infer_result_tensor)
      dur_to_numpy = time.perf_counter() - perf_c

      assert infer_result.flags.aligned

      perf_c = time.perf_counter()
      block = self._get_block(file_indices, segment_indices, infer_result)
      self._out_q.put(block)
      dur6_add_to_queue = time.perf_counter() - perf_c

      self._prediction_count += n
      self._log(
        f"Prediction made ({dur5_inference:.4} s). "
        f"Total predictions: {self._prediction_count}. Chunks: {segment_indices}"
        f"Duration half precision: {dur_half_precision:.4f} s. "
        f"Duration to numpy: {dur_to_numpy:.4f} s. "
      )

      if self._wkr_stats_queue is not None:
        wall_time = time.perf_counter() - start_time
        self._wkr_stats_queue.put(
          (
            self._pid,
            wall_time,
            dur1_wait_for_filled_slot,
            dur2_search_for_filled_slot,
            dur3_get_job,
            dur4_copy_to_device,
            dur5_inference,
            dur6_add_to_queue,
            n,
          ),
          block=False,
        )

      if self._sem_active_workers is not None:
        self._sem_active_workers.acquire(block=False)
