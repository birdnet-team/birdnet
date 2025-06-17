from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import sys
import time
from collections.abc import Generator
from logging import getLogger
from multiprocessing import shared_memory
from multiprocessing.synchronize import Semaphore
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf  # pip install soundfile

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite

from birdnet.utils import flat_sigmoid
from birdnet_v2.globals import BUSY_FLAG, DONE_FLAG, READ_FLAG, WRITE_FLAG
from birdnet_v2.inference.producer import Producer

EMPTY_ID = 0xFFFF
EMPTY_PRED = -np.inf


class Worker:
  def __init__(
    self,
    model_path: Path,
    batch_size: int,
    n_slots: int,
    prod_queue: mp.Queue,
    sem_free: mp.Semaphore,  # counts free slots
    sem_fill: mp.Semaphore,  # counts filled slots
    chunk_duration_samples,
    n_jobs: int = 4,
    top_k: int = 5,
    threshold: float = 0.03,
    whitelist: Optional[Sequence[int]] = None,
  ) -> None:
    self.top_k = top_k

    self._queue = mp.Queue()

    species_map = [f"sp_{i}" for i in range(6522)]

    valid = np.zeros(len(species_map), bool)
    if whitelist:
      valid[list(whitelist)] = True
    else:
      valid[:] = True
    valid.setflags(write=False)

    slot_ptr = mp.Value("I", 0, lock=True)  # shared memory pointer to current slot

    # workers
    self.workers = [
      mp.Process(
        target=ChildWorker(
          model_path,
          top_k,
          threshold,
          valid,
          batch_size,
          n_slots,
          chunk_duration_samples,
          slot_ptr,
          prod_queue,
          self._queue,
          sem_free,
          sem_fill,
        ),
        daemon=True,
      )
      for _ in range(n_jobs)
    ]

  @property
  def get_queue(self) -> mp.Queue:
    """
    Returns the queue used for processing audio files.
    """
    return self._queue

  def start(self) -> None:
    for p in self.workers:
      p.start()

  def join(self) -> None:
    for p in self.workers:
      p.join()
      live = sum(p.is_alive() for p in self.workers)
      print(f"WORKER - Worker {p.pid} finished. {live} workers still alive.")


class ChildWorker:
  def __init__(
    self,
    model_path: Path,
    top_k: int,
    species_thresholds: np.ndarray,
    species_blacklist: np.ndarray,
    batch_size: int,
    n_slots: int,
    chunk_duration_samples: int,
    slot_ptr: mp.Value,
    job_q: mp.Queue,
    out_q: mp.Queue,
    sem_free: Semaphore,
    sem_fill: Semaphore,
    apply_sigmoid: bool = False,
    sigmoid_sensitivity: Optional[float] = None,
    num_threads: int = 1,
  ):
    assert species_thresholds.shape[0] == 1
    assert species_blacklist.shape[0] == 1
    assert species_thresholds.shape[1] == species_blacklist.shape[1]

    self.k = top_k
    self.thresholds = species_thresholds
    self.blacklist = species_blacklist
    # Setze für ungültige Spezies den Threshold auf inf, sodass (pred >= inf) immer False ist
    self.job_q = job_q
    self.out_q = out_q
    self._slot_ptr = slot_ptr
    self.sem_free = sem_free
    self.sem_fill = sem_fill
    self.prediction_count = 0
    self.apply_sigmoid = apply_sigmoid
    self.sigmoid_sensitivity = None
    if apply_sigmoid:
      assert sigmoid_sensitivity is not None
      self.sigmoid_sensitivity = sigmoid_sensitivity

    # Interpreter
    self.interp = tflite.Interpreter(
      str(model_path.absolute()), num_threads=num_threads
    )
    self.interp.allocate_tensors()
    self.in_idx = self.interp.get_input_details()[0]["index"]
    self.out_idx = self.interp.get_output_details()[0]["index"]
    self._slot = 0

    # attatch to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not delete the shared memory access
    self._shm_file_indices = shared_memory.SharedMemory(
      name="bnet_ring_file_indices", create=False
    )
    self._shm_chunk_indices = shared_memory.SharedMemory(
      name="bnet_ring_chunk_indices", create=False
    )
    self._shm_audio_samples = shared_memory.SharedMemory(
      name="bnet_ring_audio_samples", create=False
    )
    self._shm_batch_sizes = shared_memory.SharedMemory(
      name="bnet_ring_batch_sizes", create=False
    )
    self._shm_ring_flags = shared_memory.SharedMemory(
      name="bnet_ring_flags", create=False
    )

    self._ring_file_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, self._shm_file_indices.buf
    )
    self._ring_chunk_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, self._shm_chunk_indices.buf
    )
    self._ring_audio_samples = np.ndarray(
      (n_slots, batch_size, chunk_duration_samples),
      np.float32,
      self._shm_audio_samples.buf,
    )
    self._ring_batch_sizes = np.ndarray(
      (n_slots,), np.uint16, self._shm_batch_sizes.buf
    )
    self._ring_flags = np.ndarray((n_slots,), np.uint8, self._shm_ring_flags.buf)

    self._n_slots = n_slots

    self.cached_shape: Optional[Tuple[int, int]] = (batch_size, chunk_duration_samples)
    self.interp.resize_tensor_input(self.in_idx, self.cached_shape, strict=True)
    self.interp.allocate_tensors()

  # ------------------------------------------------------------
  def _infer(self, batch: np.ndarray):
    # batch = np.ascontiguousarray(batch)

    if self.cached_shape != batch.shape:
      self.interp.resize_tensor_input(self.in_idx, batch.shape, strict=True)
      self.interp.allocate_tensors()
      self.cached_shape = batch.shape

    assert batch.flags["C_CONTIGUOUS"]
    start_time = time.perf_counter()
    self.interp.set_tensor(self.in_idx, batch)
    after_set_tensor = time.perf_counter()
    self.interp.invoke()
    final = time.perf_counter()
    logger = getLogger(__name__)
    logger.debug(
      f"WORKER({os.getpid()}) - Time to set tensor: {after_set_tensor - start_time:.4f}s, {final - after_set_tensor:.4f}s to invoke interpreter, {final - start_time:.4f}s total"
    )
    res = self.interp.get_tensor(self.out_idx)
    return res

  def _jump_to_next_slot(self) -> None:
    """Increase the slot index, wrapping around if necessary."""
    self._slot = (self._slot + 1) % self._n_slots
    assert 0 <= self._slot < self._n_slots

  def _jump_to_next_slot_ptr(self) -> None:
    """Increase the slot index, wrapping around if necessary."""
    self._slot_ptr.value = (self._slot_ptr.value + 1) % self._n_slots
    assert 0 <= self._slot_ptr.value < self._n_slots

  # ------------------------------------------------------------
  def __call__(
    self,
  ):
    logger = getLogger(__name__)
    while True:
      # job = self.job_q.get()
      # if job is None:
      #   # Stop signal
      #   logger.debug(f"WORKER({os.getpid()}) - Worker received stop signal.")
      #   self.out_q.put(None)
      #   break
      self.sem_fill.acquire()
      logger.debug(
        f"WORKER({os.getpid()}) - Worker acquired FILL; Free slots remaining: {self.sem_free}; Filled slots: {self.sem_fill}"
      )

      while True:
        with self._slot_ptr.get_lock():
          if self._ring_flags[self._slot_ptr.value] not in (READ_FLAG, DONE_FLAG):
            assert self._ring_flags[self._slot_ptr.value] in (WRITE_FLAG, BUSY_FLAG)
            self._jump_to_next_slot_ptr()
          else:
            claimed_slot = self._slot_ptr.value
            claimed_flag = self._ring_flags[claimed_slot]
            self._ring_flags[self._slot_ptr.value] = BUSY_FLAG
            self._jump_to_next_slot_ptr()
            break
      # while self._ring_flags[self._slot] not in (READ_FLAG, DONE_FLAG):
      #   self._jump_to_next_slot()
      #   logger.debug(
      #     f"WORKER({os.getpid()}) - Slot {self._slot} is not ready, looking for next available slot."
      #   )
      # slot = self._slot_ptr.value

      if claimed_flag == DONE_FLAG:
        logger.debug(
          f"WORKER({os.getpid()}) - Worker received DONE_FLAG for slot {claimed_slot}. Exiting."
        )
        self.out_q.put(None)
        break
      assert claimed_flag == READ_FLAG

      logger.debug(
        f"WORKER({os.getpid()}) - Worker acquired READ_FLAG for slot {claimed_slot}."
      )

      # slot, n = job
      n = self._ring_batch_sizes[claimed_slot]
      audio_samples = self._ring_audio_samples[claimed_slot, :n]
      file_indices = self._ring_file_indices[claimed_slot, :n]
      chunk_indices = self._ring_chunk_indices[claimed_slot, :n]
      logger.debug(
        f"WORKER({os.getpid()}) - Received job for slot {claimed_slot} with {n} samples. Chunks: {chunk_indices}"
      )
      pred = self._infer(audio_samples)

      if self.apply_sigmoid:
        assert self.sigmoid_sensitivity is not None
        pred = flat_sigmoid(
          pred,
          sensitivity=-self.sigmoid_sensitivity,
        )

      invalid_mask = filter_by_threshold(pred, self.thresholds)
      invalid_mask = combine_invalid_masks(invalid_mask, self.blacklist, in_place=True)

      # select top-k species
      top_k_species = select_top_k_indices(pred, invalid_mask, self.k)
      top_k_scores = np.take_along_axis(pred, top_k_species, axis=1)
      top_k_mask = np.take_along_axis(invalid_mask, top_k_species, axis=1)

      # sort desc by scores
      sorted_indices = get_ordered_indices(top_k_scores)
      top_k_species = np.take_along_axis(top_k_species, sorted_indices, axis=1)
      top_k_scores = np.take_along_axis(top_k_scores, sorted_indices, axis=1)
      top_k_mask = np.take_along_axis(top_k_mask, sorted_indices, axis=1)

      assert np.all(top_k_species > 0)
      assert not np.any(top_k_mask)

      self.out_q.put(
        (
          file_indices.copy(),
          chunk_indices.copy(),
          top_k_species,
          top_k_scores,
          top_k_mask,
        )
      )
      self.prediction_count += top_k_species.shape[0]
      logger.debug(
        f"WORKER({os.getpid()}) - Prediction made. Total predictions: {self.prediction_count}. Chunks: {chunk_indices}"
      )

      self._ring_flags[claimed_slot] = WRITE_FLAG
      self.sem_free.release()

      logger.debug(
        f"WORKER({os.getpid()}) - Worker released FREE. Free slots remaining: {self.sem_free}; Filled slots: {self.sem_fill}"
      )
    logger.debug(f"WORKER({os.getpid()}) - Worker finished")


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


def uint_dtype_for(max_value: int) -> np.dtype:
  """
  Return the narrowest unsigned-integer NumPy dtype that can represent
  *max_value* (inclusive).

  Examples
  --------
  >>> uint_dtype_for(100)
  dtype('uint8')
  >>> uint_dtype_for(42_000)
  dtype('uint16')
  >>> uint_dtype_for(3_000_000_000)
  dtype('uint64')
  """
  assert max_value >= 0, "max_value must be non-negative."

  for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
    if max_value <= np.iinfo(dt).max:
      return np.dtype(dt)

  raise AssertionError("Value exceeds uint64 range.")
