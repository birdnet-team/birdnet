from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.synchronize
import os
import time
from multiprocessing import Queue, shared_memory
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, Semaphore

import numpy as np
from numpy.typing import DTypeLike

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.base import AcousticInferenceBackend
from birdnet.acoustic_models.pb import AcousticPBBackend
from birdnet.acoustic_models.tf import AcousticTFBackend
from birdnet.globals import (
  DONE_FLAG,
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
)
from birdnet.helper import RingField, uint_dtype_for
from birdnet.io_lock import IOLockHandler
from birdnet.utils import flat_sigmoid_logaddexp, flat_sigmoid_logaddexp_fast


class ChildWorker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    backend_cow: AcousticInferenceBackend | None,
    top_k: int,
    species_thresholds: np.ndarray,
    species_blacklist: np.ndarray,
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_segment_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    backend_type: type[AcousticInferenceBackend],
    backend_kwargs: dict,
    segment_duration_samples: int,
    slot_ptr: Synchronized[ctypes.c_uint8]
    | Synchronized[ctypes.c_uint16]
    | Synchronized[ctypes.c_uint32]
    | Synchronized[ctypes.c_uint64],
    out_q: Queue,
    wkr_ring_access_lock: multiprocessing.synchronize.Lock,
    sem_free: Semaphore,
    sem_fill: Semaphore,
    sem_active_workers: Semaphore,
    prob_dtype: DTypeLike,
    apply_sigmoid: bool,
    sigmoid_sensitivity: float | None,
    wkr_stats_queue: mp.Queue,
    track_performance: bool,
    logging_queue: mp.Queue,
    logging_level: int,
    device: str,
    cancel_event: Event,
    io_lock_handler: IOLockHandler,
    prd_all_done_event: Event,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    assert species_thresholds.shape[0] == 1
    assert species_blacklist.shape[0] == 1
    assert species_thresholds.shape[1] == species_blacklist.shape[1]
    assert species_thresholds.flags.aligned
    assert species_blacklist.flags.aligned

    self._prd_all_done_event = prd_all_done_event
    self._wkr_ring_access_lock = wkr_ring_access_lock
    self._backend = None  # backend
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs
    self._track_performance = track_performance
    self._wkr_stats_queue = wkr_stats_queue
    self._top_k = top_k
    self._thresholds = species_thresholds
    self._blacklist = species_blacklist
    # Setze für ungültige Spezies den Threshold auf inf, sodass (pred >= inf) immer False ist
    self._out_q = out_q
    self._slot_ptr: Synchronized[int] = slot_ptr  # type: ignore
    self._sem_free = sem_free
    self._sem_filled = sem_fill
    self._sem_active_workers = sem_active_workers
    self._prediction_count = 0
    assert np.dtype(prob_dtype) in (np.float16, np.float32)
    self._prob_dtype = prob_dtype
    self._apply_sigmoid = apply_sigmoid
    self._sigmoid_sensitivity = None
    if apply_sigmoid:
      assert sigmoid_sensitivity is not None
      self._sigmoid_sensitivity = sigmoid_sensitivity
    self._io_lock_handler = io_lock_handler
    # Interpreter
    self._slot = 0
    self._batch_idx_cache = {}

    # attatch to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not delete the shared memory access
    self._n_slots = n_slots
    self._batch_size = batch_size
    self._segment_duration_samples = segment_duration_samples

    self._species_dtype: DTypeLike | None = None
    # self._cached_shape: tuple[int, ...] | None = None
    # self._model_path = str(model_path.absolute())

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

    self._lazy_init = True
    if mp.get_start_method() == "fork":
      if self._backend_type is AcousticTFBackend:
        assert backend_cow is not None
        self._init_logging()
        self._load_ring_buffers()
        self._backend = backend_cow
        self._lazy_init = False
      else:
        # PB backend does not support non lazy initialization
        assert self._backend_type is AcousticPBBackend
        assert backend_cow is None

  def _load_model(self) -> None:
    self._log_debug("Loading model...")
    try:
      self._backend = self._backend_type(**self._backend_kwargs)
      self._backend.load(self._io_lock_handler)
    except ValueError as e:
      self._log_debug(f"Failed to load model: {e}")
      raise e
    self._log_debug("Model loaded.")

  def _infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._backend is not None
    res = self._backend.infer(batch, self._device_name)
    assert res.dtype == np.float32
    res = res.astype(self._prob_dtype, copy=False)
    return res

  def _jump_to_next_slot_ptr(self) -> None:
    """Increase the slot index, wrapping around if necessary."""
    self._slot_ptr.value = (self._slot_ptr.value + 1) % self._n_slots
    assert 0 <= self._slot_ptr.value < self._n_slots

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

  def _init(self) -> bool:
    start = time.perf_counter()
    self._init_logging()
    self._load_ring_buffers()
    try:
      self._load_model()
    except ValueError:
      return False
    duration_init = time.perf_counter() - start
    self._log_debug(f"Worker {self._pid} initialized in {duration_init:.4f} seconds.")
    return True

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

  def __call__(self):
    if self._lazy_init and not self._init():
      self._cancel_event.set()
      return

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
      n_done = 0
      with self._wkr_ring_access_lock:
        for current_slot in range(self._n_slots):
          current_slot_flag = self._ring_flags[current_slot]

          # TODO: check if all ring_size slots = DONE
          if current_slot_flag == READABLE_FLAG:
            claimed_slot = current_slot
            claimed_flag = current_slot_flag
            self._ring_flags[claimed_slot] = READING_FLAG
            break
          elif current_slot_flag == DONE_FLAG:
            n_done += 1
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
        pred = self._infer(audio_samples)
      except Exception as e:
        self._log_debug(f"Error during inference: {e}")
        self._cancel_event.set()
        self._log_debug(
          f"Exiting worker {self._pid} due to error during inference. Set cancel event."
        )
        return
      dur_inference = time.perf_counter() - perf_c
      assert pred.flags.aligned

      if self._species_dtype is None:
        n_species = pred.shape[1]
        self._species_dtype = uint_dtype_for(n_species - 1)

      self._ring_flags[claimed_slot] = WRITABLE_FLAG
      self._sem_free.release()

      perf_c = time.perf_counter()

      self._log_debug(
        f"Released FREE. Free slots remaining: {self._sem_free}; Filled slots: {self._sem_filled}"
      )

      if self._apply_sigmoid:
        assert self._sigmoid_sensitivity is not None
        pred = flat_sigmoid_logaddexp_fast(
          pred,
          sensitivity=-self._sigmoid_sensitivity,
        )

      invalid_mask = (pred < self._thresholds) | self._blacklist

      shadow = np.where(invalid_mask, -np.inf, pred)

      top_k_species = np.argpartition(shadow, -self._top_k, axis=1)[
        :, -self._top_k :
      ].astype(self._species_dtype, copy=False)

      batch_idx = self._get_batch_idx(pred.shape[0])
      top_k_scores = pred[batch_idx, top_k_species]
      top_k_mask = invalid_mask[batch_idx, top_k_species]

      self._out_q.put(
        (
          file_indices,
          segment_indices,
          top_k_species,
          top_k_scores,
          top_k_mask,
        )
      )
      dur_add_to_queue = time.perf_counter() - perf_c

      self._prediction_count += top_k_species.shape[0]
      self._log_debug(
        f"Prediction made ({dur_inference:.4} s). Total predictions: {self._prediction_count}. Chunks: {segment_indices}"
      )

      if self._track_performance:
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

      self._sem_active_workers.acquire(block=False)

    self._log_debug("Finished.")
    self._uninit()

  def _get_batch_idx(self, batch_size: int) -> np.ndarray:
    if batch_size not in self._batch_idx_cache:
      self._batch_idx_cache[batch_size] = np.arange(batch_size)[:, None]
    return self._batch_idx_cache[batch_size]


def filter_by_threshold(
  logits: np.ndarray,
  thresh_vec: np.ndarray,
) -> np.ndarray:
  assert logits.ndim == 2
  assert thresh_vec.ndim == 2
  # logits: (N, C)
  # thresh_vec: (1, C)  or  (N, C)
  assert logits.shape[1] == thresh_vec.shape[1]
  invalid_full = logits < thresh_vec
  return invalid_full


def combine_invalid_masks(
  mask_a: np.ndarray,  # (N, C)   bool
  mask_b: np.ndarray,  # (1, C) o. (N, C)
  *,
  in_place: bool = False,
) -> np.ndarray:
  assert mask_a.ndim == 2 and mask_b.ndim == 2
  assert mask_a.shape[1] == mask_b.shape[1]

  if in_place:
    np.logical_or(mask_a, mask_b, out=mask_a)
    return mask_a
  else:
    result = np.logical_or(mask_a, mask_b)
    return result


def select_top_k_indices(
  logits: np.ndarray,
  invalid_mask: np.ndarray,
  k: int,
) -> np.ndarray:
  assert logits.ndim == 2
  assert logits.shape == invalid_mask.shape
  assert k > 0
  n_species = logits.shape[1]
  idx_dtype = uint_dtype_for(n_species - 1)

  shadow = np.where(invalid_mask, -np.inf, logits)

  idx = np.argpartition(shadow, -k, axis=1)[:, -k:]
  idx = idx.astype(idx_dtype, copy=False)
  return idx


def get_ordered_indices(
  scores: np.ndarray,  # (N, k) float32
) -> np.ndarray:
  assert scores.ndim == 2
  order = np.argsort(-scores, axis=1)
  return order
