import logging
import multiprocessing
import multiprocessing as mp
import os
from collections.abc import Generator, Iterable
from itertools import count, islice
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue, shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from numpy.typing import DTypeLike
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm

import birdnet_v2.logging_utils as bn_logging
from birdnet.types import Species, TimeInterval
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  get_chunks_with_overlap,
  itertools_batched,
  resample_array,
)
from birdnet_v2.globals import DONE_FLAG, READ_FLAG, WRITE_FLAG
from birdnet_v2.helper import (
  RingField,
  get_max_n_chunks,
  max_value_for_uint_dtype,
  uint_dtype_for,
)


def get_chunks_with_overlap(
  total_duration_s: Union[int, float],
  chunk_duration_s: Union[int, float],
  overlap_duration_s: Union[int, float],
) -> Generator[Tuple[float, float], None, None]:
  assert total_duration_s > 0
  assert chunk_duration_s > 0
  assert 0 <= overlap_duration_s < chunk_duration_s

  if not isinstance(overlap_duration_s, float):
    overlap_duration_s = float(overlap_duration_s)
  if not isinstance(chunk_duration_s, float):
    chunk_duration_s = float(chunk_duration_s)
  if not isinstance(total_duration_s, float):
    total_duration_s = float(total_duration_s)

  step_duration = chunk_duration_s - overlap_duration_s
  for start in count(0.0, step_duration):
    assert start < total_duration_s
    if (end := start + chunk_duration_s) < total_duration_s:
      yield start, end
    else:
      yield start, total_duration_s
      break


def resample_array(
  x: npt.NDArray, sample_rate: int, target_sample_rate: int
) -> npt.NDArray:
  assert len(x.shape) == 1
  assert sample_rate > 0
  assert target_sample_rate > 0

  if sample_rate == target_sample_rate:
    return x

  target_sample_count = round(len(x) / sample_rate * target_sample_rate)
  x_resampled: npt.NDArray = resample(x, target_sample_count)
  assert x_resampled.dtype == x.dtype
  return x_resampled


class Producer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    files: List[Path],
    batch_size: int,
    n_slots: int,
    n_jobs: int,
    rf_file_indices: RingField,
    rf_chunk_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    sem_free_slots: Semaphore,  # counts free slots
    sem_filled_slots: Semaphore,  # counts filled slots
    max_chunk_idx_ptr: mp.RawValue,
    logging_queue: mp.Queue,
    logging_level: int,
    chunk_duration_s: float = 3.0,
    overlap_duration_s: float = 0.0,
    target_sample_rate: int = 48000,
    use_bandpass: bool = False,
    bandpass_fmin: Optional[int] = None,
    bandpass_fmax: Optional[int] = None,
    fmin: Optional[int] = None,
    fmax: Optional[int] = None,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    self.chunk_duration_s = chunk_duration_s
    self.overlap_duration_s = overlap_duration_s
    self.target_sample_rate = target_sample_rate
    self._batch_size = batch_size
    self._n_jobs = n_jobs
    self._n_slots = n_slots
    self._sem_free_slots = sem_free_slots
    self._sem_filled_slots = sem_filled_slots
    self._slot = 0
    self._files = files
    self._use_bandpass = use_bandpass
    self._max_chunk_idx_ptr = max_chunk_idx_ptr

    if use_bandpass:
      assert bandpass_fmin is not None
      assert bandpass_fmax is not None
      assert 0 <= bandpass_fmin < bandpass_fmax <= target_sample_rate // 2
      self.bandpass_fmin = bandpass_fmin
      self.bandpass_fmax = bandpass_fmax
      self.sig_fmin = fmin
      self.sig_fmax = fmax
    else:
      self.bandpass_fmin = None
      self.bandpass_fmax = None
      self.sig_fmin = None
      self.sig_fmax = None

    self.chunk_duration_samples = target_sample_rate * int(chunk_duration_s)

    self._rf_file_indices = rf_file_indices
    self._rf_chunk_indices = rf_chunk_indices
    self._rf_audio_samples = rf_audio_samples
    self._rf_batch_sizes = rf_batch_sizes
    self._rf_flags = rf_flags

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

    self._max_supported_chunk_index = (
      max_value_for_uint_dtype(rf_chunk_indices.dtype) - 1
    )

  def _load_ring_buffers(self) -> None:
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

  def _uninit(self) -> None:
    self._uninit_logging()

  def get_chunks_from_files(
    self,
  ) -> Generator[tuple[int, int, npt.NDArray[np.float32]], None, None]:
    for file_index, path in enumerate(self._files):
      audio_duration = get_audio_duration(path)
      file_n_chunks = get_max_n_chunks(
        audio_duration, self.chunk_duration_s, self.overlap_duration_s
      )
      file_max_chunk_index = file_n_chunks - 1

      if file_max_chunk_index > self._max_chunk_idx_ptr.value:
        if file_max_chunk_index > self._max_supported_chunk_index:
          self._logger.error(
            f"File {path} has a duration of {audio_duration / 60:.2f} min and contains {file_n_chunks} chunks, which exceeds the maximum supported amount of chunks {self._max_supported_chunk_index + 1}. Please set maximum audio duration."
          )
          continue
        self._max_chunk_idx_ptr.value = file_max_chunk_index
      chunks = load_audio_in_chunks_with_overlap(
        path,
        chunk_duration_s=self.chunk_duration_s,
        overlap_duration_s=self.overlap_duration_s,
        target_sample_rate=self.target_sample_rate,
      )

      # fill last chunk with silence up to chunksize if it is smaller than 3s
      chunks = (
        fillup_with_silence(chunk, self.chunk_duration_samples) for chunk in chunks
      )

      if self._use_bandpass:
        assert self.bandpass_fmin is not None
        assert self.bandpass_fmax is not None
        assert self.sig_fmin is not None
        assert self.sig_fmax is not None

        chunks = (
          bandpass_signal(
            chunk,
            self.target_sample_rate,
            self.bandpass_fmin,
            self.bandpass_fmax,
            self.sig_fmin,
            self.sig_fmax,
          )
          for chunk in chunks
        )

      for chunk_index, chunk in enumerate(chunks):
        yield file_index, chunk_index, chunk

  def __call__(self) -> None:
    self._init()
    buffer_input = self.get_chunks_from_files()
    for batch in itertools_batched(buffer_input, self._batch_size):
      file_indices, chunk_indices, audio_samples = zip(*batch)
      max_chunk_index = max(chunk_indices)
      if max_chunk_index > self._max_supported_chunk_index:
        self._logger.error(
          f"Chunk index {max_chunk_index} exceeds maximum supported chunk index {self._max_supported_chunk_index}. Please set maximum audio duration. Cancelling proceessing."
        )
        break

      self._flush_batch(file_indices, chunk_indices, audio_samples)

    # send poison pills
    for _ in range(self._n_jobs):
      self._set_done_flag()

    self._uninit()

  def _jump_to_next_slot(self) -> None:
    """Increase the slot index, wrapping around if necessary."""
    self._slot = (self._slot + 1) % self._n_slots
    assert 0 <= self._slot < self._n_slots

  def _set_done_flag(self) -> None:
    """Set the DONE_FLAG in the shared memory to signal that no more data will be produced."""
    self._sem_free_slots.acquire()
    self._logger.debug(
      f"PRODUCER - Producer acquired FREE. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
    )
    while self._ring_flags[self._slot] != WRITE_FLAG:
      self._jump_to_next_slot()

    assert 0 <= self._slot < self._n_slots
    self._ring_flags[self._slot] = DONE_FLAG
    self._logger.debug(f"PRODUCER - Set DONE_FLAG on slot {self._slot}.")
    self._jump_to_next_slot()
    self._sem_filled_slots.release()

  def _flush_batch(self, file_indices, chunk_indices, audio_samples) -> None:
    self._sem_free_slots.acquire()
    self._logger.debug(
      f"PRODUCER - Producer acquired FREE. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
    )
    while self._ring_flags[self._slot] != WRITE_FLAG:
      self._jump_to_next_slot()

    current_batch_size = len(audio_samples)
    assert len(file_indices) == current_batch_size
    assert len(chunk_indices) == current_batch_size
    assert 0 <= self._slot < self._n_slots
    assert current_batch_size <= self._batch_size
    self._ring_file_indices[self._slot, :current_batch_size] = np.asarray(
      file_indices, self._ring_file_indices.dtype
    )
    # TODO könnte man noch bei den anderen auch machen
    assert max(chunk_indices) < max_value_for_uint_dtype(self._ring_chunk_indices.dtype)
    assert min(chunk_indices) >= 0
    self._ring_chunk_indices[self._slot, :current_batch_size] = np.asarray(
      chunk_indices, self._ring_chunk_indices.dtype
    )

    self._ring_audio_samples[self._slot, :current_batch_size] = np.asarray(
      np.stack(audio_samples, 0), self._ring_audio_samples.dtype
    )
    self._ring_batch_sizes[self._slot] = current_batch_size

    # self._queue.put((slot, current_batch_size))
    self._logger.debug(
      f"PRODUCER - Flushed batch to shared memory on slot {self._slot}, batch size {current_batch_size}. Chunk indices: {chunk_indices}"
    )

    self._ring_flags[self._slot] = READ_FLAG
    self._jump_to_next_slot()
    self._sem_filled_slots.release()
    self._logger.debug(
      f"PRODUCER - Producer released FILL. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
    )


def get_audio_duration(audio_path: Path) -> float:
  """
  Returns the duration of the audio file in seconds.
  """
  assert audio_path.is_file()
  sf_info = sf.info(audio_path)
  result = float(sf_info.duration)
  return result


def load_audio_in_chunks_with_overlap(
  audio_path: Path,
  /,
  *,
  chunk_duration_s: float = 3,
  overlap_duration_s: float = 0,
  # read_duration_s: Optional[float] = None,
  target_sample_rate: int = 48000,
) -> Generator[npt.NDArray[np.float32], None, None]:
  assert audio_path.is_file()

  sf_info = sf.info(audio_path)
  is_mono = sf_info.channels == 1
  assert is_mono

  sample_rate = sf_info.samplerate

  timestamps = get_chunks_with_overlap(
    float(sf_info.duration),
    float(chunk_duration_s),
    float(overlap_duration_s),
  )

  for start, end in timestamps:
    start_samples = round(start * sample_rate)
    end_samples = round(end * sample_rate)
    audio, _ = sf.read(
      audio_path, start=start_samples, stop=end_samples, dtype=np.float32
    )
    audio = resample_array(audio, sample_rate, target_sample_rate)
    yield audio


import atexit
import contextlib


@contextlib.contextmanager
def shm_ring_from_name(name: str, size: int):
  shm = SharedMemory(create=True, name=name, size=size)
  try:
    yield shm
  finally:
    shm.close()
    shm.unlink()  # wird sogar bei CTRL-C im finally ausgeführt
    print(f"Shared memory {name} cleaned up.")


def test_producing():
  import faulthandler
  import sys

  faulthandler.enable(file=sys.stderr, all_threads=True)

  audio_path = Path("test-dataset/test_dataset_1x1440min/0.wav")
  audio_path = Path("example/soundscape.wav")
  n_files = 2
  files = [audio_path] * n_files

  batch_size = 3
  n_slots = 8
  n_workers = 4
  prod_queue = Queue()
  sem_free = mp.Semaphore(n_slots)
  sem_fill = mp.Semaphore()
  chunk_duration_s = 3
  overlap_duration_s = 0
  target_sample_rate = 48000
  chunk_duration_samples = int(chunk_duration_s * target_sample_rate)

  with (
    shm_ring_from_name(
      "bnet_ring_file_indices",
      n_slots * batch_size * np.dtype(np.uint32).itemsize,
    ) as shm_file_indices,
    shm_ring_from_name(
      "bnet_ring_chunk_indices", n_slots * batch_size * np.dtype(np.uint32).itemsize
    ) as shm_chunk_indices,
    shm_ring_from_name(
      "bnet_ring_audio_samples",
      n_slots * batch_size * chunk_duration_samples * np.dtype(np.float32).itemsize,
    ) as shm_audio_samples,
  ):
    print("Shared memory initialized.")

    prod = mp.Process(
      target=Producer(
        files,
        batch_size,
        n_slots,
        n_workers,
        prod_queue,
        sem_free,
        sem_fill,
        chunk_duration_s,
        overlap_duration_s,
        target_sample_rate,
      ),
      daemon=True,
    )
    prod.start()

    ring_file_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, shm_file_indices.buf
    )
    ring_chunk_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, shm_chunk_indices.buf
    )
    ring_audio_samples = np.ndarray(
      (n_slots, batch_size, chunk_duration_samples),
      np.float32,
      shm_audio_samples.buf,
    )

    res = []
    while True:
      job = prod_queue.get()
      if job is None:
        break
      slot, n = job
      sem_fill.acquire()
      file_indices = ring_file_indices[slot, :n]
      chunk_indices = ring_chunk_indices[slot, :n]
      audio_samples = ring_audio_samples[slot, :n]
      res.append(list(chunk_indices))
      sem_free.release()
  print("Producer finished processing.")
  print(res)


def test_chunking():
  from time import perf_counter

  t1 = perf_counter()
  res = list(
    load_audio_in_chunks_with_overlap(Path("test-dataset/test_dataset_1x1440min/0.wav"))
  )
  print(f"Loaded {len(res)} chunks in {perf_counter() - t1:.2f} seconds")


if __name__ == "__main__":
  test_producing()
