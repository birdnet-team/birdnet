from __future__ import annotations

import ctypes
import multiprocessing.synchronize
import os
import time
from collections.abc import Generator
from itertools import count
from multiprocessing import Queue, shared_memory
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

import birdnet.logging_utils as bn_logging
from birdnet.globals import (
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
)
from birdnet.helper import (
  SF_FORMATS,
  RingField,
  get_max_n_segments,
  max_value_for_uint_dtype,
)
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  itertools_batched,
)

# def get_segments_with_overlap(
#   total_duration_s: int | float,
#   segment_duration_s: int | float,
#   overlap_duration_s: int | float,
# ) -> Generator[tuple[float, float], None, None]:
#   assert total_duration_s > 0
#   assert segment_duration_s > 0
#   assert 0 <= overlap_duration_s < segment_duration_s

#   if not isinstance(overlap_duration_s, float):
#     overlap_duration_s = float(overlap_duration_s)
#   if not isinstance(segment_duration_s, float):
#     segment_duration_s = float(segment_duration_s)
#   if not isinstance(total_duration_s, float):
#     total_duration_s = float(total_duration_s)

#   step_duration = segment_duration_s - overlap_duration_s
#   for start in count(0.0, step=step_duration):
#     assert start < total_duration_s
#     if (end := start + segment_duration_s) < total_duration_s:
#       yield start, end
#     else:
#       yield start, total_duration_s
#       break


def get_segments_with_overlap_all(
  total_duration_s: int | float,
  segment_duration_s: int | float,
  overlap_duration_s: int | float,
) -> Generator[tuple[float, float], None, None]:
  assert total_duration_s > 0
  assert segment_duration_s > 0
  assert 0 <= overlap_duration_s < segment_duration_s

  if not isinstance(overlap_duration_s, float):
    overlap_duration_s = float(overlap_duration_s)
  if not isinstance(segment_duration_s, float):
    segment_duration_s = float(segment_duration_s)
  if not isinstance(total_duration_s, float):
    total_duration_s = float(total_duration_s)

  step_duration = segment_duration_s - overlap_duration_s
  for start in count(0.0, step=step_duration):
    if start >= total_duration_s:
      break
    end = min(start + segment_duration_s, total_duration_s)
    yield start, end


def resample_array(
  x: npt.NDArray, sample_rate: int, target_sample_rate: int
) -> npt.NDArray:
  assert len(x.shape) == 1
  assert sample_rate > 0
  assert target_sample_rate > 0

  if sample_rate == target_sample_rate:
    return x

  target_sample_count = round(len(x) / sample_rate * target_sample_rate)

  from scipy.signal import resample

  x_resampled: npt.NDArray = resample(x, target_sample_count)
  assert x_resampled.dtype == x.dtype
  return x_resampled


class Producer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    files_queue: Queue,
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_segment_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    sem_free_slots: Semaphore,
    sem_filled_slots: Semaphore,
    max_segment_idx_ptr: ctypes.c_uint8
    | ctypes.c_uint16
    | ctypes.c_uint32
    | ctypes.c_uint64,
    prod_done_ptr: Synchronized[ctypes.c_uint8]
    | Synchronized[ctypes.c_uint16]
    | Synchronized[ctypes.c_uint32]
    | Synchronized[ctypes.c_uint64],
    n_feeders: int,
    prd_ring_access_lock: multiprocessing.synchronize.Lock,
    logging_queue: Queue,
    logging_level: int,
    prod_stats_queue: Queue | None,
    segment_duration_s: float,
    overlap_duration_s: float,
    target_sample_rate: int,
    cancel_event: Event,
    prd_all_done_event: Event,
    use_bandpass: bool,
    bandpass_fmin: int,
    bandpass_fmax: int,
    fmin: int | None,
    fmax: int | None,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    self._prd_all_done_event = prd_all_done_event
    self._prd_ring_access_lock = prd_ring_access_lock
    self._prod_stats_queue = prod_stats_queue
    self._segment_duration_s = segment_duration_s
    self._overlap_duration_s = overlap_duration_s
    self._target_sample_rate = target_sample_rate
    self._batch_size = batch_size
    self._n_slots = n_slots
    self._sem_free_slots = sem_free_slots
    self._sem_filled_slots = sem_filled_slots
    self._files_queue = files_queue
    self._use_bandpass = use_bandpass
    self._max_segment_idx_ptr = max_segment_idx_ptr  # type: ignore
    self._prod_done_ptr: Synchronized[int] = prod_done_ptr  # type: ignore
    self._n_producers = n_feeders

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

    self.segment_duration_samples = target_sample_rate * int(segment_duration_s)

    self._rf_file_indices = rf_file_indices
    self._rf_segment_indices = rf_segment_indices
    self._rf_audio_samples = rf_audio_samples
    self._rf_batch_sizes = rf_batch_sizes
    self._rf_flags = rf_flags

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

    self._max_supported_segment_index = (
      max_value_for_uint_dtype(rf_segment_indices.dtype) - 1
    )

    self._cancel_event = cancel_event

  def _load_ring_buffers(self) -> None:
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

  def _init(self) -> None:
    self._init_logging()
    self._load_ring_buffers()
    self._logger.debug(f"PRODUCER({os.getpid()}) - Initialized.")

  def _uninit(self) -> None:
    self._logger.debug(f"PRODUCER({os.getpid()}) - Uninitializing...")
    self._uninit_logging()

  def get_segments_from_files(
    self,
  ) -> Generator[tuple[int, int, npt.NDArray[np.float32]], None, None]:
    while True:
      if self._check_cancel_event():
        break

      queue_entry = self._files_queue.get(block=True)
      poison_pill = queue_entry is None
      if poison_pill:
        self._logger.debug(f"PRODUCER({os.getpid()}) - Received poison pill. Exiting.")
        break
      assert isinstance(queue_entry, tuple)
      file_index, path = queue_entry

      audio_duration_s = get_audio_duration_s(path)
      file_n_segments = get_max_n_segments(
        audio_duration_s, self._segment_duration_s, self._overlap_duration_s
      )
      file_max_segment_index = file_n_segments - 1

      if file_max_segment_index > self._max_segment_idx_ptr.value:
        if file_max_segment_index > self._max_supported_segment_index:
          self._logger.error(
            f"File {path} has a duration of {audio_duration_s / 60:.2f} min and contains {file_n_segments} segments, which exceeds the maximum supported amount of segments {self._max_supported_segment_index + 1}. Please set maximum audio duration."
          )
          continue
        self._max_segment_idx_ptr.value = file_max_segment_index
      segments = load_audio_in_segments_with_overlap(
        path,
        segment_duration_s=self._segment_duration_s,
        overlap_duration_s=self._overlap_duration_s,
        target_sample_rate=self._target_sample_rate,
      )

      # fill last segment with silence up to segmentsize if it is smaller than 3s
      segments = (
        fillup_with_silence(segment, self.segment_duration_samples)
        for segment in segments
      )

      if self._use_bandpass:
        assert self.bandpass_fmin is not None
        assert self.bandpass_fmax is not None
        assert self.sig_fmin is not None
        assert self.sig_fmax is not None

        segments = (
          bandpass_signal(
            segment,
            self._target_sample_rate,
            self.bandpass_fmin,
            self.bandpass_fmax,
            self.sig_fmin,
            self.sig_fmax,
          )
          for segment in segments
        )

      for segment_index, segment in enumerate(segments):
        yield file_index, segment_index, segment

  @property
  def _pid(self) -> int:
    return os.getpid()

  def _iter_files(self) -> None:
    start_time = time.perf_counter()
    if self._check_cancel_event():
      return

    buffer_input = self.get_segments_from_files()
    batches = itertools_batched(buffer_input, self._batch_size)

    while True:
      perf_c = time.perf_counter()
      while not self._sem_free_slots.acquire(timeout=1.0):
        if self._check_cancel_event():
          return
      wait_time_for_free_slot = time.perf_counter() - perf_c

      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Producer acquired FREE. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
      )

      if self._check_cancel_event():
        return

      perf_c = time.perf_counter()
      try:
        batch = next(batches)
      except StopIteration:
        self._logger.debug(
          f"PRODUCER({os.getpid()}) - No more batches to process. Exiting."
        )
        self._sem_free_slots.release()
        break
      batch_loading_duration = time.perf_counter() - perf_c

      file_indices, segment_indices, audio_samples = zip(*batch, strict=False)
      max_segment_index = max(segment_indices)
      if max_segment_index > self._max_supported_segment_index:
        self._logger.error(
          f"Chunk index {max_segment_index} exceeds maximum supported segment index {self._max_supported_segment_index}. Please set maximum audio duration. Cancelling proceessing."
        )
        self._cancel_event.set()
        return

      assert self._ring_flags is not None

      claimed_flag = None
      claimed_slot = None

      perf_c = time.perf_counter()
      with self._prd_ring_access_lock:
        for current_slot in range(self._n_slots):
          current_slot_flag = self._ring_flags[current_slot]
          if current_slot_flag == WRITABLE_FLAG:
            claimed_slot = current_slot
            claimed_flag = current_slot_flag
            if claimed_flag == WRITABLE_FLAG:
              self._ring_flags[claimed_slot] = WRITING_FLAG
              break
          else:
            assert current_slot_flag in (
              READABLE_FLAG,
              READING_FLAG,
              WRITING_FLAG,
            )
      free_slot_search_time = time.perf_counter() - perf_c

      if claimed_slot is None:
        raise AssertionError(
          "No free slot found in the ring buffer but sem_free was available!"
        )
      assert claimed_flag == WRITABLE_FLAG

      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Acquired WRITING_FLAG for slot {claimed_slot}."
      )

      if self._check_cancel_event():
        return

      perf_c = time.perf_counter()
      self._flush_batch(claimed_slot, file_indices, segment_indices, audio_samples)
      flush_duration = time.perf_counter() - perf_c

      self._ring_flags[claimed_slot] = READABLE_FLAG
      self._sem_filled_slots.release()

      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Producer released FILL. Free slots remaining: {self._sem_free_slots}; Filled slots: {self._sem_filled_slots}"
      )

      if self._prod_stats_queue is not None:
        now = time.perf_counter()
        process_total_duration = now - start_time
        n = len(file_indices)
        self._prod_stats_queue.put(
          (
            self._pid,
            process_total_duration,
            batch_loading_duration,
            wait_time_for_free_slot,
            free_slot_search_time,
            flush_duration,
            n,
          )
        )

    self._logger.debug(
      f"PRODUCER({os.getpid()}) - Finished processing files. Total time: {time.perf_counter() - start_time:.2f} seconds."
    )

  def __call__(self) -> None:
    self._init()

    self._iter_files()

    if self._check_cancel_event():
      return

    with self._prod_done_ptr:
      self._prod_done_ptr.value = self._prod_done_ptr.value + 1
      self._logger.debug(
        f"PRODUCER({os.getpid()}) - Set prod_done_ptr to {self._prod_done_ptr.value}."
      )
      is_last_producer = self._prod_done_ptr.value == self._n_producers

    if is_last_producer:
      self._logger.debug(f"PRODUCER({os.getpid()}) - Last producer finished.")
      self._prd_all_done_event.set()
      assert self._files_queue.qsize() == 0

  def _check_cancel_event(self) -> bool:
    if self._cancel_event.is_set():
      self._logger.debug(f"PRODUCER({os.getpid()}) - Received cancel event.")
      return True
    return False

  def _flush_batch(
    self, claimed_slot, file_indices, segment_indices, audio_samples
  ) -> None:
    assert self._ring_audio_samples is not None
    assert self._ring_file_indices is not None
    assert self._ring_segment_indices is not None
    assert self._ring_batch_sizes is not None

    current_batch_size = len(audio_samples)
    assert len(file_indices) == current_batch_size
    assert len(segment_indices) == current_batch_size
    assert 0 <= claimed_slot < self._n_slots
    assert current_batch_size <= self._batch_size
    self._ring_file_indices[claimed_slot, :current_batch_size] = np.asarray(
      file_indices, self._ring_file_indices.dtype
    )
    # TODO könnte man noch bei den anderen auch machen
    assert max(segment_indices) < max_value_for_uint_dtype(
      self._ring_segment_indices.dtype
    )
    assert min(segment_indices) >= 0
    self._ring_segment_indices[claimed_slot, :current_batch_size] = np.asarray(
      segment_indices, self._ring_segment_indices.dtype
    )

    self._ring_audio_samples[claimed_slot, :current_batch_size] = np.asarray(
      np.stack(audio_samples, 0), self._ring_audio_samples.dtype
    )
    self._ring_batch_sizes[claimed_slot] = current_batch_size

    self._logger.debug(
      f"PRODUCER({os.getpid()}) - Flushed batch to shared memory on slot {claimed_slot}, batch size {current_batch_size}. Chunk indices: {segment_indices}"
    )


def get_audio_duration_s(audio_path: Path) -> float:
  """
  Returns the duration of the audio file in seconds.
  """
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS
  sf_info = sf.info(audio_path)
  result = float(sf_info.duration)
  return result


def load_audio_in_segments_with_overlap(
  audio_path: Path,
  /,
  *,
  segment_duration_s: float = 3,
  overlap_duration_s: float = 0,
  # read_duration_s: Optional[float] = None,
  target_sample_rate: int = 48000,
) -> Generator[npt.NDArray[np.float32], None, None]:
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS

  sf_info = sf.info(audio_path)

  sample_rate = sf_info.samplerate

  timestamps = get_segments_with_overlap_all(
    float(sf_info.duration),
    float(segment_duration_s),
    float(overlap_duration_s),
  )

  for start, end in timestamps:
    start_samples = round(start * sample_rate)
    end_samples = round(end * sample_rate)
    audio, _ = sf.read(
      audio_path, start=start_samples, stop=end_samples, dtype="float32"
    )

    if audio.ndim == 2:
      n_channels = audio.shape[1]
      assert n_channels > 1
      audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = resample_array(audio, sample_rate, target_sample_rate)
    yield audio
