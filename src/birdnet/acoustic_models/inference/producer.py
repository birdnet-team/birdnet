from __future__ import annotations

import ctypes
import os
from collections import deque
from collections.abc import Generator
from itertools import count
from multiprocessing import Queue, shared_memory
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample

import birdnet.logging_utils as bn_logging
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  get_chunks_with_overlap,
  itertools_batched,
  resample_array,
)
from birdnet.globals import (
  DONE_FLAG,
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
)
from birdnet.helper import (
  SF_FORMATS,
  RingField,
  get_max_n_chunks,
  max_value_for_uint_dtype,
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


class ChildProducer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    files_queue: Queue,
    slot_ptr: Synchronized[ctypes.c_uint8]
    | Synchronized[ctypes.c_uint16]
    | Synchronized[ctypes.c_uint32]
    | Synchronized[ctypes.c_uint64],
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_chunk_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    sem_free_slots: Semaphore,
    sem_filled_slots: Semaphore,
    max_chunk_idx_ptr: ctypes.c_uint8
    | ctypes.c_uint16
    | ctypes.c_uint32
    | ctypes.c_uint64,
    prod_done_ptr: Synchronized[ctypes.c_uint8]
    | Synchronized[ctypes.c_uint16]
    | Synchronized[ctypes.c_uint32]
    | Synchronized[ctypes.c_uint64],
    n_prods: int,
    logging_queue: Queue,
    logging_level: int,
    chunk_duration_s: float,
    overlap_duration_s: float,
    target_sample_rate: int,
    cancel_event: Event,
    use_bandpass: bool,
    bandpass_fmin: int | None,
    bandpass_fmax: int | None,
    fmin: int | None,
    fmax: int | None,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    self.chunk_duration_s = chunk_duration_s
    self.overlap_duration_s = overlap_duration_s
    self.target_sample_rate = target_sample_rate
    self._batch_size = batch_size
    self._n_slots = n_slots
    self._sem_free_slots = sem_free_slots
    self._sem_filled_slots = sem_filled_slots
    self._slot_ptr: Synchronized[int] = slot_ptr  # type: ignore
    self._files_queue = files_queue
    self._use_bandpass = use_bandpass
    self._max_chunk_idx_ptr = max_chunk_idx_ptr  # type: ignore
    self._prod_done_ptr: Synchronized[int] = prod_done_ptr  # type: ignore
    self._n_producers = n_prods

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

    self._max_supported_chunk_index = (
      max_value_for_uint_dtype(rf_chunk_indices.dtype) - 1
    )

    self._cancel_event = cancel_event

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
    self._logger.debug(f"PRODUCER({os.getpid()}) - Initialized.")

  def _uninit(self) -> None:
    self._logger.debug(f"PRODUCER({os.getpid()}) - Uninitializing...")
    self._uninit_logging()

  def get_chunks_from_files(
    self,
  ) -> Generator[tuple[int, int, npt.NDArray[np.float32]], None, None]:
    while True:
      queue_entry = self._files_queue.get()
      poison_pill = queue_entry is None
      if poison_pill:
        self._logger.debug(f"PRODUCER({os.getpid()}) - Received poison pill. Exiting.")
        break
      assert isinstance(queue_entry, tuple)
      file_index, path = queue_entry

      audio_duration_s = get_audio_duration_s(path)
      file_n_chunks = get_max_n_chunks(
        audio_duration_s, self.chunk_duration_s, self.overlap_duration_s
      )
      file_max_chunk_index = file_n_chunks - 1

      if file_max_chunk_index > self._max_chunk_idx_ptr.value:
        if file_max_chunk_index > self._max_supported_chunk_index:
          self._logger.error(
            f"File {path} has a duration of {audio_duration_s / 60:.2f} min and contains {file_n_chunks} chunks, which exceeds the maximum supported amount of chunks {self._max_supported_chunk_index + 1}. Please set maximum audio duration."
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

      cancel = False
      while True:
        try:
          self._sem_free_slots.acquire(timeout=1.0)
          break
        except TimeoutError:
          if self._cancel_event.is_set():
            cancel = True
            break

      if self._cancel_event.is_set():
        cancel = True

      if cancel:
        self._logger.debug(f"PRODUCER({os.getpid()}) - Cancel event set. Exiting.")
        self._uninit()
        return

      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Producer acquired FREE. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
      )

      self._flush_batch(file_indices, chunk_indices, audio_samples)

      self._sem_filled_slots.release()
      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Producer released FILL. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
      )

    with self._prod_done_ptr.get_lock():
      self._prod_done_ptr.value = self._prod_done_ptr.value + 1
      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Set prod_done_ptr to {self._prod_done_ptr.value}."
      )
      is_last_producer = self._prod_done_ptr.value == self._n_producers

    if is_last_producer:
      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Last producer finished. Sending poison pills."
      )
      # send poison pills
      # can also use n_jobs here, but this is faster
      for _ in range(self._n_slots):
        self._set_done_flag()

    self._uninit()

  def _jump_to_next_slot_ptr(self) -> None:
    """Increase the slot index, wrapping around if necessary."""
    self._slot_ptr.value = (self._slot_ptr.value + 1) % self._n_slots
    assert 0 <= self._slot_ptr.value < self._n_slots

  def _set_done_flag(self) -> None:
    assert self._ring_flags is not None
    # only one producer process gets into this method

    """Set the DONE_FLAG in the shared memory to signal that no more data will be produced."""
    self._sem_free_slots.acquire()
    self._logger.debug(
      f"PRODUCER({os.getpid()}) - Producer acquired FREE. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
    )
    while self._ring_flags[self._slot_ptr.value] != WRITABLE_FLAG:
      self._jump_to_next_slot_ptr()

    assert 0 <= self._slot_ptr.value < self._n_slots
    self._ring_flags[self._slot_ptr.value] = DONE_FLAG
    self._logger.debug(
      f"PRODUCER({os.getpid()}) - Set DONE_FLAG on slot {self._slot_ptr.value}."
    )
    self._jump_to_next_slot_ptr()
    self._sem_filled_slots.release()

  def _flush_batch(self, file_indices, chunk_indices, audio_samples) -> None:
    assert self._ring_audio_samples is not None
    assert self._ring_file_indices is not None
    assert self._ring_chunk_indices is not None
    assert self._ring_batch_sizes is not None
    assert self._ring_flags is not None

    cancel = False
    claimed_flag = None
    claimed_slot = None

    while True:
      if self._cancel_event.is_set():
        cancel = True
        break

      with self._slot_ptr.get_lock():
        current_slot = self._slot_ptr.value
        current_slot_flag = self._ring_flags[current_slot]
        # can never be DONE because this flag is set after all chunks from all files have been flushed
        assert current_slot_flag != DONE_FLAG

        if current_slot_flag == WRITABLE_FLAG:
          claimed_slot = current_slot
          claimed_flag = current_slot_flag
          if claimed_flag == WRITABLE_FLAG:
            self._ring_flags[claimed_slot] = WRITING_FLAG
          self._jump_to_next_slot_ptr()
          break
        else:
          assert current_slot_flag in (
            READABLE_FLAG,
            READING_FLAG,
            WRITING_FLAG,
          )
          self._jump_to_next_slot_ptr()

    if self._cancel_event.is_set():
      cancel = True

    if cancel:
      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Cancel event set. Exiting _flush_batch."
      )
      return

    assert claimed_flag is not None
    assert claimed_slot is not None

    assert claimed_flag == WRITABLE_FLAG
    self._logger.debug(
      f"PRODUCER({os.getpid()}) - Acquired WRITABLE_FLAG for slot {claimed_slot}."
    )

    current_batch_size = len(audio_samples)
    assert len(file_indices) == current_batch_size
    assert len(chunk_indices) == current_batch_size
    assert 0 <= claimed_slot < self._n_slots
    assert current_batch_size <= self._batch_size
    self._ring_file_indices[claimed_slot, :current_batch_size] = np.asarray(
      file_indices, self._ring_file_indices.dtype
    )
    # TODO könnte man noch bei den anderen auch machen
    assert max(chunk_indices) < max_value_for_uint_dtype(self._ring_chunk_indices.dtype)
    assert min(chunk_indices) >= 0
    self._ring_chunk_indices[claimed_slot, :current_batch_size] = np.asarray(
      chunk_indices, self._ring_chunk_indices.dtype
    )

    self._ring_audio_samples[claimed_slot, :current_batch_size] = np.asarray(
      np.stack(audio_samples, 0), self._ring_audio_samples.dtype
    )
    self._ring_batch_sizes[claimed_slot] = current_batch_size

    self._logger.debug(
      f"PRODUCER({os.getpid()}) - Flushed batch to shared memory on slot {claimed_slot}, batch size {current_batch_size}. Chunk indices: {chunk_indices}"
    )

    self._ring_flags[claimed_slot] = READABLE_FLAG


def get_audio_duration_s(audio_path: Path) -> float:
  """
  Returns the duration of the audio file in seconds.
  """
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS
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
  assert audio_path.suffix.upper() in SF_FORMATS

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
