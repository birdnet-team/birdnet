from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory
from multiprocessing.synchronize import Semaphore
from pathlib import Path

import numpy as np
import soundfile as sf  # pip install soundfile
from numpy.typing import DTypeLike

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite

import birdnet_v2.logging_utils as bn_logging
from birdnet.utils import flat_sigmoid
from birdnet_v2.acoustic_models.base import AcousticInferenceBackend
from birdnet_v2.globals import (
  DONE_FLAG,
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
)
from birdnet_v2.helper import RingField, uint_dtype_for


class ChildWorker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    model_path: Path,
    top_k: int,
    species_thresholds: np.ndarray,
    species_blacklist: np.ndarray,
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_chunk_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    backend: AcousticInferenceBackend,
    chunk_duration_samples: int,
    slot_ptr: mp.Value,
    out_q: mp.Queue,
    sem_free: Semaphore,
    sem_fill: Semaphore,
    prob_dtype: DTypeLike,
    apply_sigmoid: bool,
    sigmoid_sensitivity: float | None,
    pred_dur_queue: mp.SimpleQueue,
    track_performance: bool,
    logging_queue: mp.Queue,
    logging_level: int,
    num_threads: int = 1,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    assert species_thresholds.shape[0] == 1
    assert species_blacklist.shape[0] == 1
    assert species_thresholds.shape[1] == species_blacklist.shape[1]

    self._backend = backend
    self._track_performance = track_performance
    self._pred_dur_queue = pred_dur_queue
    self._top_k = top_k
    self._thresholds = species_thresholds
    self._blacklist = species_blacklist
    # Setze für ungültige Spezies den Threshold auf inf, sodass (pred >= inf) immer False ist
    self._out_q = out_q
    self._slot_ptr = slot_ptr
    self._sem_free = sem_free
    self._sem_filled = sem_fill
    self._prediction_count = 0
    assert np.dtype(prob_dtype) in (np.float16, np.float32)
    self._prob_dtype = prob_dtype
    self._apply_sigmoid = apply_sigmoid
    self._sigmoid_sensitivity = None
    if apply_sigmoid:
      assert sigmoid_sensitivity is not None
      self._sigmoid_sensitivity = sigmoid_sensitivity

    # Interpreter
    self._interp: tflite.Interpreter | None = None
    self._slot = 0

    # attatch to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not delete the shared memory access
    self._n_slots = n_slots
    self._batch_size = batch_size
    self._chunk_duration_samples = chunk_duration_samples
    # self._cached_shape: tuple[int, ...] | None = None
    # self._model_path = str(model_path.absolute())
    self._num_threads = num_threads

    self._rf_file_indices = rf_file_indices
    self._rf_chunk_indices = rf_chunk_indices
    self._rf_audio_samples = rf_audio_samples
    self._rf_batch_sizes = rf_batch_sizes
    self._rf_flags = rf_flags

    self._in_idx: int | None = None
    self._out_idx: int | None = None

    self._shm_file_indices: shared_memory.SharedMemory | None = None
    self._shm_chunk_indices: shared_memory.SharedMemory | None = None
    self._shm_audio_samples: shared_memory.SharedMemory | None = None
    self._shm_batch_sizes: shared_memory.SharedMemory | None = None
    self._shm_ring_flags: shared_memory.SharedMemory | None = None

    self._ring_file_indices: np.ndarray | None = None
    self._ring_chunk_indices: np.ndarray | None = None
    self._ring_audio_samples: np.ndarray | None = None
    self._ring_batch_sizes: np.ndarray | None = None
    self._ring_flags: np.ndarray | None = None
    # self._mm: mmap.mmap | None = None

  def _load_model(self):
    self._backend.lazy_load()
    # assert self._interp is None

    # # memory_map not working for TF 2.15.1:
    # # f = open(self._model_path, "rb")
    # # self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # self._interp = tflite.Interpreter(self._model_path, num_threads=self._num_threads)
    # self._interp.allocate_tensors()
    # self._in_idx = self._interp.get_input_details()[0]["index"]
    # self._out_idx = self._interp.get_output_details()[0]["index"]

  # def _set_tensor(self, batch: np.ndarray):
  #   assert self._interp is not None
  #   assert batch.flags["C_CONTIGUOUS"]
  #   assert batch.ndim == 2

  #   shape = batch.shape
  #   if self._cached_shape != shape:
  #     self._interp.resize_tensor_input(self._in_idx, shape, strict=True)
  #     self._interp.allocate_tensors()
  #     self._cached_shape = shape
  #   self._interp.set_tensor(self._in_idx, batch)

  # # ------------------------------------------------------------
  def _infer(self, batch: np.ndarray):
    start_time = time.perf_counter()
    res = self._backend.infer(batch)
    if self._track_performance:
      final = time.perf_counter()
      pred_dur = final - start_time
      self._pred_dur_queue.put((pred_dur, batch.shape[0]))
    assert res.dtype == np.float32
    res = res.astype(self._prob_dtype, copy=False)
    return res

  # def _infer_old(self, batch: np.ndarray):
  #   assert self._interp is not None

  #   self._set_tensor(batch)
  #   start_time = time.perf_counter()
  #   self._interp.invoke()
  #   if self._track_performance:
  #     final = time.perf_counter()
  #     pred_dur = final - start_time
  #     self._pred_dur_queue.put((pred_dur, batch.shape[0]))
  #   res: np.ndarray = self._interp.get_tensor(self._out_idx)
  #   assert res.dtype == np.float32

  #   res = res.astype(self._prob_dtype, copy=False)
  #   return res

  def _jump_to_next_slot_ptr(self) -> None:
    """Increase the slot index, wrapping around if necessary."""
    self._slot_ptr.value = (self._slot_ptr.value + 1) % self._n_slots
    assert 0 <= self._slot_ptr.value < self._n_slots

  def _load_ring_buffers(self) -> None:
    # attach to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not delete the shared memory access
    self._shm_file_indices, self._ring_file_indices = (
      self._rf_file_indices.attach_and_get_array()
    )
    self._shm_chunk_indices, self._ring_chunk_indices = (
      self._rf_chunk_indices.attach_and_get_array()
    )
    self._shm_audio_samples, self._ring_audio_samples = (
      self._rf_audio_samples.attach_and_get_array()
    )
    self._shm_batch_sizes, self._ring_batch_sizes = (
      self._rf_batch_sizes.attach_and_get_array()
    )
    self._shm_ring_flags, self._ring_flags = self._rf_flags.attach_and_get_array()

  def _init(self) -> None:
    self._init_logging()
    self._load_ring_buffers()
    self._load_model()

  def _uninit(self) -> None:
    self._uninit_logging()

  @property
  def _pid(self) -> int:
    return os.getpid()

  def _log_debug(self, msg: str) -> None:
    self._logger.debug(f"WORKER({self._pid}) - {msg}")

  def __call__(self):
    self._init()

    assert self._ring_flags is not None
    assert self._ring_file_indices is not None
    assert self._ring_chunk_indices is not None
    assert self._ring_audio_samples is not None
    assert self._ring_batch_sizes is not None

    while True:
      self._sem_filled.acquire()
      self._log_debug(
        f"Acquired FILL; Free slots remaining: {self._sem_free}; Filled slots: {self._sem_filled}"
      )

      while True:
        with self._slot_ptr.get_lock():
          current_slot = self._slot_ptr.value
          current_slot_flag = self._ring_flags[current_slot]
          # TODO: check if all ring_size slots = DONE
          if current_slot_flag in (READABLE_FLAG, DONE_FLAG):
            claimed_slot = current_slot
            claimed_flag = current_slot_flag
            if claimed_flag == READABLE_FLAG:
              self._ring_flags[claimed_slot] = READING_FLAG
            self._jump_to_next_slot_ptr()
            break
          else:
            assert current_slot_flag in (
              WRITABLE_FLAG,
              WRITING_FLAG,
              READING_FLAG,
            )
            self._jump_to_next_slot_ptr()

      if claimed_flag == DONE_FLAG:
        self._log_debug(f"Received DONE_FLAG for slot {claimed_slot}. Exiting.")
        self._out_q.put(None)
        break
      assert claimed_flag == READABLE_FLAG

      self._log_debug(f"Acquired READ_FLAG for slot {claimed_slot}.")

      n = self._ring_batch_sizes[claimed_slot]
      audio_samples = self._ring_audio_samples[claimed_slot, :n]
      file_indices = self._ring_file_indices[claimed_slot, :n]
      chunk_indices = self._ring_chunk_indices[claimed_slot, :n]
      self._log_debug(
        f"Received job for slot {claimed_slot} with {n} chunks: {chunk_indices}"
      )
      pred = self._infer(audio_samples)

      if self._apply_sigmoid:
        assert self._sigmoid_sensitivity is not None
        pred = flat_sigmoid(
          pred,
          sensitivity=-self._sigmoid_sensitivity,
        )

      invalid_mask = filter_by_threshold(pred, self._thresholds)
      invalid_mask = combine_invalid_masks(invalid_mask, self._blacklist, in_place=True)

      # select top-k species
      top_k_species = select_top_k_indices(pred, invalid_mask, self._top_k)
      top_k_scores = np.take_along_axis(pred, top_k_species, axis=1)
      top_k_mask = np.take_along_axis(invalid_mask, top_k_species, axis=1)

      # sort desc by scores
      sorted_indices = get_ordered_indices(top_k_scores)
      top_k_species = np.take_along_axis(top_k_species, sorted_indices, axis=1)
      top_k_scores = np.take_along_axis(top_k_scores, sorted_indices, axis=1)
      top_k_mask = np.take_along_axis(top_k_mask, sorted_indices, axis=1)

      self._out_q.put(
        (
          file_indices.copy(),
          chunk_indices.copy(),
          top_k_species,
          top_k_scores,
          top_k_mask,
        )
      )
      self._prediction_count += top_k_species.shape[0]
      self._log_debug(
        f"Prediction made. Total predictions: {self._prediction_count}. Chunks: {chunk_indices}"
      )

      self._ring_flags[claimed_slot] = WRITABLE_FLAG
      self._sem_free.release()

      self._log_debug(
        f"Released FREE. Free slots remaining: {self._sem_free}; Filled slots: {self._sem_filled}"
      )
    self._log_debug("Finished.")
    self._uninit()


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
