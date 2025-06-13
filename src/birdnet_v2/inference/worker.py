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
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf  # pip install soundfile

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite

from birdnet.utils import flat_sigmoid
from birdnet_v2.inference.producer import Producer

EMPTY_ID = 0  # 0xFFFF
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
      print(f"Worker {p.pid} finished. {live} workers still alive.")


class ChildWorker:
  def __init__(
    self,
    model_path: Path,
    top_k: int,
    thresh: np.ndarray,
    valid: np.ndarray,
    batch_size: int,
    n_slots: int,
    chunk_duration_samples: int,
    job_q: mp.Queue,
    out_q: mp.Queue,
    sem_free: mp.Semaphore,
    sem_fill: mp.Semaphore,
    apply_sigmoid: bool = False,
    sigmoid_sensitivity: Optional[float] = None,
  ):
    self.k = top_k
    # Setze für ungültige Spezies den Threshold auf 1.0, sodass (pred >= 1.0) in der Regel fehlschlägt.
    self.thresh = np.where(valid, thresh, 1.0)[np.newaxis, :]
    self.job_q = job_q
    self.out_q = out_q
    self.sem_free = sem_free
    self.sem_fill = sem_fill
    self.apply_sigmoid = apply_sigmoid
    self.sigmoid_sensitivity = None
    if apply_sigmoid:
      assert sigmoid_sensitivity is not None
      self.sigmoid_sensitivity = sigmoid_sensitivity

    # Interpreter
    self.interp = tflite.Interpreter(str(model_path), num_threads=1)
    self.interp.allocate_tensors()
    self.in_idx = self.interp.get_input_details()[0]["index"]
    self.out_idx = self.interp.get_output_details()[0]["index"]

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
    print(
      f"Time to set tensor: {after_set_tensor - start_time:.4f}s, {final - after_set_tensor:.4f}s to invoke interpreter, {final - start_time:.4f}s total"
    )
    res = self.interp.get_tensor(self.out_idx)
    return res

  # ------------------------------------------------------------
  def __call__(
    self,
  ):
    logger = getLogger(__name__)
    while True:
      job = self.job_q.get()
      if job is None:
        # Stop signal
        logger.info("Worker received stop signal.")
        self.out_q.put(None)
        break
      slot, n = job
      self.sem_fill.acquire()
      audio_samples = self._ring_audio_samples[slot, :n]
      file_indices = self._ring_file_indices[slot, :n]
      chunk_indices = self._ring_chunk_indices[slot, :n]
      pred = self._infer(audio_samples)

      if self.apply_sigmoid:
        assert self.sigmoid_sensitivity is not None
        pred = flat_sigmoid(
          pred,
          sensitivity=-self.sigmoid_sensitivity,
        )

      # Filterung vor der Top-k-Auswahl:
      # Erzeuge eine Maske für gültige Vorhersagen (Threshold und gültige Spezies)
      valid_mask = pred >= self.thresh
      # Wenn möglich, in-place modifizieren (sparen einer Kopie)
      pred[~valid_mask] = EMPTY_PRED

      # Auswahl der Top-k Werte pro Zeile anhand der gefilterten Vorhersagen
      species_indices = np.argpartition(pred, -self.k, axis=1)[:, -self.k :]
      species_probs = np.take_along_axis(pred, species_indices, axis=1)
      order = np.argsort(-species_probs, axis=1)
      row = np.arange(n)[:, None]
      species_indices = species_indices[row, order]
      species_probs = species_probs[row, order]

      # Setze alle -∞-Werte, welche nicht den Filter passiert haben, auf EMPTY_ID und 0.0
      pred_mask = species_probs == EMPTY_PRED
      species_indices[pred_mask] = EMPTY_ID

      species_indices = species_indices.astype(np.uint16, copy=False)
      species_probs = species_probs.astype(np.float16, copy=False)

      self.out_q.put(
        (
          file_indices,
          chunk_indices,
          species_indices,
          species_probs,
        )
      )
      self.sem_free.release()
    logger.info("Worker finished")
