from __future__ import annotations

import ctypes
import multiprocessing.synchronize
import os
import time
from collections.abc import Callable, Generator
from functools import partial
from itertools import count
from multiprocessing import Queue, shared_memory
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from queue import Empty

import numpy as np
import numpy.typing as npt
import soundfile

import birdnet.acoustic_models.inference_pipeline.logs as bn_logging
from birdnet.globals import (
  READABLE_FLAG,
  READING_FLAG,
  WRITABLE_FLAG,
  WRITING_FLAG,
  Float32Array,
  FloatArray,
  IntArray,
)
from birdnet.helper import (
  SF_FORMATS,
  apply_speed_to_samples,
  assert_queue_is_empty,
  duration_as_samples,
  get_n_segments_speed,
  max_value_for_uint_dtype,
)
from birdnet.shm import RingField
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  itertools_batched,
)


def get_segments_with_overlap_all_int(
  total_duration: int,
  segment_duration: int,
  overlap_duration: int,
) -> Generator[tuple[int, int], None, None]:
  assert isinstance(total_duration, int)
  assert isinstance(segment_duration, int)
  assert isinstance(overlap_duration, int)
  assert total_duration > 0
  assert segment_duration > 0
  assert 0 <= overlap_duration < segment_duration

  step_duration = segment_duration - overlap_duration
  for start in count(0, step=step_duration):
    if start >= total_duration:
      break
    end = min(start + segment_duration, total_duration)
    yield start, end


def calculate_target_sample_count(
  n_samples: int, sample_rate: int, target_sample_rate: int
) -> int:
  assert sample_rate > 0
  assert target_sample_rate > 0
  x, y = divmod(n_samples, sample_rate)
  if y != 0:
    raise ValueError("original_sample_count must be a multiple of sample_rate")
  target_sample_count = x * target_sample_rate
  return target_sample_count


def resample_array_by_sr(
  array: npt.NDArray, sample_rate: int, target_sample_rate: int
) -> npt.NDArray:
  assert len(array.shape) == 1
  assert sample_rate > 0
  assert target_sample_rate > 0

  if sample_rate == target_sample_rate:
    return array

  dur_seconds = len(array) / sample_rate
  target_sample_count = round(dur_seconds * target_sample_rate)

  from scipy.signal import resample

  array_resampled: npt.NDArray = resample(array, target_sample_count)
  assert array_resampled.dtype == array.dtype
  return array_resampled


def resample_array_by_stretching(
  array: npt.NDArray, target_n_samples: int
) -> npt.NDArray:
  assert len(array.shape) == 1

  if len(array) == target_n_samples:
    return array

  from scipy.signal import resample

  array_resampled: npt.NDArray = resample(array, target_n_samples)
  assert array_resampled.dtype == array.dtype
  return array_resampled


class Producer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    session_id: str,
    input_queue: Queue,
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
    end_event: Event,
    start_signal: Event,
    n_feeders: int,
    prd_ring_access_lock: multiprocessing.synchronize.Lock,
    logging_queue: Queue,
    logging_level: int,
    prod_stats_queue: Queue | None,
    segment_duration_s: float,
    overlap_duration_s: float,
    speed: float,
    target_sample_rate: int,
    cancel_event: Event,
    all_finished: Event,
    use_bandpass: bool,
    bandpass_fmin: int,
    bandpass_fmax: int,
    fmin: int | None,
    fmax: int | None,
    unprocessed_inputs_queue: Queue,
  ):
    super().__init__(session_id, __name__, logging_queue, logging_level)
    self._end_event = end_event
    self._all_finished = all_finished
    self._prd_ring_access_lock = prd_ring_access_lock
    self._prod_stats_queue = prod_stats_queue
    self._segment_duration_s = segment_duration_s
    self._overlap_duration_s = overlap_duration_s
    self._target_sample_rate = target_sample_rate
    self._speed = speed
    self._batch_size = batch_size
    self._n_slots = n_slots
    self._sem_free_slots = sem_free_slots
    self._sem_filled_slots = sem_filled_slots
    self._input_queue = input_queue
    self._use_bandpass = use_bandpass
    self._max_segment_idx_ptr = max_segment_idx_ptr  # type: ignore
    self._prod_done_ptr: Synchronized[int] = prod_done_ptr  # type: ignore
    self._n_producers = n_feeders
    self._start_signal = start_signal
    self._unprocessable_inputs: set[int] = set()
    self._unprocessed_inputs_queue = unprocessed_inputs_queue

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

  def get_segments_from_input(
    self, input_idx: int, inp_data: Path | tuple[np.ndarray, int]
  ) -> Generator[tuple[int, npt.NDArray[np.float32]], None, None]:
    audio_n_samples: int = 0
    audio_sample_rate: int = 0

    if isinstance(inp_data, Path):
      read_file_successfully = False
      try:
        sf_info = get_sf_info(inp_data)
        audio_n_samples = get_audio_n_samples_from_sf(sf_info)
        audio_sample_rate = get_sample_rate_from_sf(sf_info)
        read_file_successfully = True
      except (
        soundfile.LibsndfileError,
        soundfile.SoundFileRuntimeError,
        soundfile.SoundFileError,
      ) as error:
        self._logger.warning(
          f"FA_{os.getpid()}: "
          f"Could not read audio file #{input_idx} '{inp_data.absolute()}': {error}. "
          f"Skipping file.",
        )
      except Exception as error:
        self._logger.warning(
          f"FA_{os.getpid()}: "
          f"Could not read audio file #{input_idx} '{inp_data.absolute()}': {error}. "
          f"Skipping file.",
          exc_info=error,
          stack_info=True,
        )

      if not read_file_successfully:
        self._unprocessable_inputs.add(input_idx)
        return
    else:
      assert isinstance(inp_data, tuple)
      assert len(inp_data) == 2
      assert isinstance(inp_data[0], np.ndarray)
      assert isinstance(inp_data[1], int)
      audio_array, audio_sample_rate = inp_data
      audio_n_samples = audio_array.shape[0]

    assert audio_sample_rate > 0

    audio_duration_s = audio_n_samples / audio_sample_rate

    file_n_segments = get_n_segments_speed(
      audio_duration_s, self._segment_duration_s, self._overlap_duration_s, self._speed
    )
    file_max_segment_index = file_n_segments - 1

    if file_max_segment_index > self._max_segment_idx_ptr.value:
      if file_max_segment_index > self._max_supported_segment_index:
        inp_path = (
          f"#{input_idx} '{inp_data.absolute()}'"
          if isinstance(inp_data, Path)
          else f"<in-memory audio array> at index {input_idx}"
        )
        self._logger.error(
          f"Input {inp_path} has a duration of {audio_duration_s / 60:.2f} min and "
          f"contains {file_n_segments} segments, which exceeds the maximum supported "
          f"amount of segments {self._max_supported_segment_index + 1}. "
          f"Please set maximum audio duration."
        )
        return
      self._max_segment_idx_ptr.value = file_max_segment_index

    segments: Generator[npt.NDArray[np.float32], None, None]
    if isinstance(inp_data, Path):
      try:
        segments = get_file_segments_with_overlap(
          inp_data,
          audio_n_samples=audio_n_samples,
          sample_rate=audio_sample_rate,
          segment_duration_s=self._segment_duration_s,
          overlap_duration_s=self._overlap_duration_s,
          speed=self._speed,
          target_sample_rate=self._target_sample_rate,
        )
      except (
        soundfile.LibsndfileError,
        soundfile.SoundFileRuntimeError,
        soundfile.SoundFileError,
      ) as error:
        self._logger.warning(
          f"FA_{os.getpid()}: "
          f"Could not read file #{input_idx} '{inp_data.absolute()}': {error}. "
          f"Skipping file.",
        )
        self._unprocessable_inputs.add(input_idx)
        return
      except ValueError as error:
        self._logger.warning(
          f"FA_{os.getpid()}: "
          f"Could not process file #{input_idx} '{inp_data.absolute()}': {error}. "
          f"Skipping file.",
        )
        self._unprocessable_inputs.add(input_idx)
        return
      except Exception as error:
        self._logger.warning(
          f"FA_{os.getpid()}: "
          f"Could not process file #{input_idx} '{inp_data.absolute()}': {error}. "
          f"Skipping file.",
          exc_info=error,
          stack_info=True,
        )
        self._unprocessable_inputs.add(input_idx)
        return
    else:
      assert isinstance(inp_data, tuple)
      assert len(inp_data) == 2
      assert isinstance(inp_data[0], np.ndarray)
      assert isinstance(inp_data[1], int)

      audio_array, audio_sample_rate = inp_data
      try:
        segments = get_data_segments_with_overlap(
          audio_array,
          audio_sample_rate,
          segment_duration_s=self._segment_duration_s,
          overlap_duration_s=self._overlap_duration_s,
          speed=self._speed,
          target_sample_rate=self._target_sample_rate,
        )
      except ValueError as error:
        self._logger.warning(
          f"FA_{os.getpid()}: Could not process data at index {input_idx}: {error}. "
          f"Skipping it.",
        )
        self._unprocessable_inputs.add(input_idx)
        return

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

    yield from enumerate(segments)

  def get_segments_from_files(
    self,
  ) -> Generator[tuple[int, int, npt.NDArray[np.float32]], None, None]:
    while True:
      if self._check_cancel_event():
        return

      while True:
        try:
          input_queue_entry = self._input_queue.get(block=True, timeout=1.0)
          break
        except Empty:
          if self._check_cancel_event():
            return

      poison_pill = input_queue_entry is None
      if poison_pill:
        self._log("Received poison pill. Exiting.")
        break
      assert isinstance(input_queue_entry, tuple)
      input_index, inp_data = input_queue_entry

      for segment_index, segment in self.get_segments_from_input(input_index, inp_data):
        yield input_index, segment_index, segment

  @property
  def _pid(self) -> int:
    return os.getpid()

  def _iter_files(self) -> None:
    start_time = time.perf_counter()
    buffer_input = self.get_segments_from_files()

    batches = itertools_batched(buffer_input, self._batch_size)

    while True:
      perf_c = time.perf_counter()
      while not self._sem_free_slots.acquire(timeout=1.0):
        if self._check_cancel_event():
          return
      wait_time_for_free_slot = time.perf_counter() - perf_c

      self._log(
        f"Producer acquired FREE. Free slots remaining: {self._sem_free_slots}; "
        f"Filled slots: {self._sem_filled_slots}"
      )

      if self._check_cancel_event():
        return

      perf_c = time.perf_counter()
      try:
        batch = next(batches)
      except StopIteration:
        self._log("No more batches to process. Exiting.")
        self._sem_free_slots.release()
        break
      batch_loading_duration = time.perf_counter() - perf_c

      b_file_indices, b_segment_indices, b_audio_samples = zip(*batch, strict=False)
      max_segment_index = max(b_segment_indices)
      if max_segment_index > self._max_supported_segment_index:
        self._logger.error(
          f"Chunk index {max_segment_index} exceeds maximum supported segment index "
          f"{self._max_supported_segment_index}. Please set maximum audio duration. "
          f"Cancelling proceessing."
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

      self._log(f"Acquired WRITING_FLAG for slot {claimed_slot}.")

      if self._check_cancel_event():
        return

      perf_c = time.perf_counter()
      self._write_batch_to_slot(
        claimed_slot, b_file_indices, b_segment_indices, b_audio_samples
      )
      flush_duration = time.perf_counter() - perf_c

      self._ring_flags[claimed_slot] = READABLE_FLAG
      self._sem_filled_slots.release()

      self._log(
        "Producer released FILL. "
        f"Free slots remaining: {self._sem_free_slots}; "
        f"Filled slots: {self._sem_filled_slots}"
      )

      if self._prod_stats_queue is not None:
        now = time.perf_counter()
        process_total_duration = now - start_time
        n = len(b_file_indices)
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

    self._unprocessed_inputs_queue.put(self._unprocessable_inputs, block=True)
    self._unprocessable_inputs = set()

    self._log(
      "Finished processing files. "
      f"Total time: {time.perf_counter() - start_time:.2f} seconds."
    )

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

  def _write_batch_to_slot(
    self,
    claimed_slot: int,
    batch_file_indices: tuple[int, ...],
    batch_segment_indices: tuple[int, ...],
    batch_audio_samples: tuple[int, ...],
  ) -> None:
    assert self._ring_audio_samples is not None
    assert self._ring_file_indices is not None
    assert self._ring_segment_indices is not None
    assert self._ring_batch_sizes is not None

    current_batch_size = len(batch_audio_samples)
    assert len(batch_file_indices) == current_batch_size
    assert len(batch_segment_indices) == current_batch_size
    assert 0 <= claimed_slot < self._n_slots
    assert current_batch_size <= self._batch_size
    self._ring_file_indices[claimed_slot, :current_batch_size] = np.asarray(
      batch_file_indices, self._ring_file_indices.dtype
    )
    # TODO könnte man noch bei den anderen auch machen
    assert max(batch_segment_indices) < max_value_for_uint_dtype(
      self._ring_segment_indices.dtype
    )
    assert min(batch_segment_indices) >= 0
    self._ring_segment_indices[claimed_slot, :current_batch_size] = np.asarray(
      batch_segment_indices, self._ring_segment_indices.dtype
    )

    self._ring_audio_samples[claimed_slot, :current_batch_size] = np.asarray(
      np.stack(batch_audio_samples, 0), self._ring_audio_samples.dtype
    )
    self._ring_batch_sizes[claimed_slot] = current_batch_size

    self._log(
      f"Flushed batch to shared memory on slot {claimed_slot}, "
      f"batch size {current_batch_size}. Chunk indices: {batch_segment_indices}"
    )

  def _log(self, message: str) -> None:
    self._logger.debug(f"P_{os.getpid()}: {message}")

  def __call__(self) -> None:
    self._init_logging()

    try:
      self._load_ring_buffers()

      self._run_main_loop()
    except Exception as e:
      self._logger.exception(
        "Producer encountered an exception.", exc_info=e, stack_info=True
      )
      self._cancel_event.set()

    self._uninit_logging()

  def _run_main_loop(self) -> None:
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
      self._run_main()

  def _run_main(self) -> None:
    if self._check_cancel_event():
      return

    self._iter_files()
    self._log("Itered all files.")

    if self._check_cancel_event():
      return

    with self._prod_done_ptr:
      self._prod_done_ptr.value = self._prod_done_ptr.value + 1
      self._log(f"Set prod_done_ptr to {self._prod_done_ptr.value}.")
      is_last_producer = self._prod_done_ptr.value == self._n_producers

    if is_last_producer:
      self._log("Last producer finished.")
      self._all_finished.set()
      assert_queue_is_empty(self._input_queue)


def get_sf_info(audio_path: Path) -> soundfile._SoundFileInfo:
  """
  Returns the soundfile info of the audio file.
  """
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS
  sf_info = soundfile.info(audio_path)
  return sf_info


def get_audio_duration_from_sf(sf_info: soundfile._SoundFileInfo) -> float:
  result = float(sf_info.duration)
  return result


def get_audio_n_samples_from_sf(sf_info: soundfile._SoundFileInfo) -> int:
  result = int(sf_info.frames)
  return result


def get_sample_rate_from_sf(sf_info: soundfile._SoundFileInfo) -> int:
  result = int(sf_info.samplerate)
  return result


def get_audio_duration_s(audio_path: Path) -> float:
  """
  Returns the duration of the audio file in seconds.
  """
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS
  sf_info = soundfile.info(audio_path)
  result = float(sf_info.duration)
  return result


def get_audio_n_samples(audio_path: Path) -> int:
  """
  Returns the number of samples in the audio file.
  """
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS
  sf_info = soundfile.info(audio_path)
  result = int(sf_info.frames)
  return result


def get_file_segments_with_overlap(
  audio_path: Path,
  audio_n_samples: int,
  sample_rate: int,
  segment_duration_s: float,
  overlap_duration_s: float,
  speed: float,
  target_sample_rate: int,
) -> Generator[npt.NDArray[np.float32], None, None]:
  """Load audio in overlapping segments with optional speed change.

  speed:
    Speed factor for audio playback. Values < 1.0 slow down the audio,
    values > 1.0 speed it up.

  segment_duration_s, overlap_duration_s:
    Refer to the *speed-adjusted* playback domain. For example, with
    segment_duration_s=3 and target_sample_rate=48000, each yielded
    segment will have 3 * 48000 samples, independent of the speed
    setting. Changing speed only changes how many segments are produced.
  """
  read_method = partial(read_file_in_mono, audio_path=audio_path)

  segments = get_segments_with_overlap(
    audio_n_samples,
    sample_rate,
    read_method,
    segment_duration_s=segment_duration_s,
    overlap_duration_s=overlap_duration_s,
    speed=speed,
    target_sample_rate=target_sample_rate,
  )

  yield from segments


def get_data_segments_with_overlap(
  audio_array: npt.NDArray,
  sample_rate: int,
  segment_duration_s: float,
  overlap_duration_s: float,
  speed: float,
  target_sample_rate: int,
) -> Generator[npt.NDArray[np.float32], None, None]:
  audio_n_samples = audio_array.shape[0]
  read_method = partial(read_data_in_mono, audio_data=audio_array)

  segments = get_segments_with_overlap(
    audio_n_samples,
    sample_rate,
    read_method,
    segment_duration_s=segment_duration_s,
    overlap_duration_s=overlap_duration_s,
    speed=speed,
    target_sample_rate=target_sample_rate,
  )

  yield from segments


def get_segments_with_overlap(
  audio_n_samples: int,
  audio_sr: int,
  audio_read_fn: Callable[[int, int], npt.NDArray[np.float32]],
  segment_duration_s: float,
  overlap_duration_s: float,
  speed: float,
  target_sample_rate: int,
) -> Generator[npt.NDArray[np.float32], None, None]:
  assert audio_sr > 0
  assert audio_n_samples >= 0
  assert target_sample_rate > 0

  if audio_n_samples == 0:
    return

  n_samples_orig_seg = duration_as_samples(segment_duration_s, audio_sr)
  n_samples_orig_overlap = duration_as_samples(overlap_duration_s, audio_sr)

  if not 1 / round(segment_duration_s * audio_sr) <= speed:
    raise ValueError(
      f"for sample rate {audio_sr} and segment duration of {segment_duration_s} "
      f"parameter speed must be >= {1 / round(segment_duration_s * audio_sr)}"
    )

  if not speed <= audio_n_samples:
    raise ValueError(
      f"for audio with {audio_n_samples} samples parameter speed must be <= "
      f"{audio_n_samples}"
    )

  for (
    start_samples_scaled,
    end_samples_scaled,
    target_n_samples,
  ) in get_segments_with_overlap_samples(
    audio_n_samples,
    segment_samples=n_samples_orig_seg,
    overlap_samples=n_samples_orig_overlap,
    speed=speed,
  ):
    audio = audio_read_fn(start_samples_scaled, end_samples_scaled)
    assert audio.dtype == np.float32

    audio = resample_array_by_stretching(audio, target_n_samples)
    audio = resample_array_by_sr(audio, audio_sr, target_sample_rate)
    yield audio


def get_segments_with_overlap_samples(
  n_samples: int,
  segment_samples: int,
  overlap_samples: int,
  speed: float = 1.0,
) -> Generator[tuple[int, int, int], None, None]:
  """
  returns tuples of (start_samples_scaled, end_samples_scaled, target_n_samples)
  samples lie in range [0, n_samples]
  target_n_samples is the number of samples after speed adjustment
  """
  assert 1 / segment_samples <= speed <= n_samples
  # assert speed >= 0.01

  n_samples_orig = n_samples
  n_samples_orig_seg = segment_samples
  n_samples_orig_overlap = overlap_samples
  # n_samples_orig_scaled = round(n_samples_orig / speed)
  n_samples_orig_scaled = apply_speed_to_samples(n_samples_orig, 1 / speed)

  # Playback duration after speed adjustment: slower -> longer, faster -> shorter.
  # n_samples_scaled = round(n_samples_orig * speed)
  n_samples_scaled_seg = apply_speed_to_samples(segment_samples, speed)
  n_samples_scaled_overlap = apply_speed_to_samples(overlap_samples, speed)

  timestamps_orig_samples = get_segments_with_overlap_all_int(
    n_samples_orig_scaled,
    n_samples_orig_seg,
    n_samples_orig_overlap,
  )

  timestamps_scaled_samples = get_segments_with_overlap_all_int(
    n_samples_orig,
    n_samples_scaled_seg,
    n_samples_scaled_overlap,
  )

  for (start_samples_orig, end_samples_orig), (
    start_samples_scaled,
    end_samples_scaled,
  ) in zip(timestamps_orig_samples, timestamps_scaled_samples, strict=True):
    assert start_samples_orig < n_samples_orig_scaled
    assert end_samples_orig <= n_samples_orig_scaled
    assert start_samples_scaled < n_samples_orig
    assert end_samples_scaled <= n_samples_orig

    target_n_samples = end_samples_orig - start_samples_orig

    yield start_samples_scaled, end_samples_scaled, target_n_samples


def read_data_in_mono(
  start_samples: int,
  end_samples: int,
  audio_data: npt.NDArray,
) -> npt.NDArray[np.float32]:
  assert 0 <= start_samples < end_samples <= audio_data.shape[0]
  if (
    not isinstance(audio_data, np.ndarray)
    and audio_data.dtype != np.float32
    and audio_data.ndim not in (1, 2)
  ):
    raise AssertionError("Invalid audio data provided.")
  audio = audio_data[start_samples:end_samples]
  audio = to_float32(audio)
  return convert_to_mono(audio)


def to_float32(audio: IntArray | FloatArray) -> Float32Array:
  """
  Convert integer or floating audio arrays to float32.
  """
  if np.issubdtype(audio.dtype, np.floating):
    # type: ignore because numpy can't prove cast is correct
    return np.asarray(audio, dtype=np.float32)

  if np.issubdtype(audio.dtype, np.integer):
    info = np.iinfo(audio.dtype)  # type: ignore
    # scale e.g., int 16: −32768 ... 32767 -> take 32768
    scale = max(abs(info.min), abs(info.max))
    return audio.astype(np.float32) / float(scale)

  raise ValueError(f"Unsupported dtype: {audio.dtype}")


def read_file_in_mono(
  start_samples: int,
  end_samples: int,
  audio_path: Path,
) -> npt.NDArray[np.float32]:
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS
  assert 0 <= start_samples < end_samples

  audio, _ = soundfile.read(
    audio_path, start=start_samples, stop=end_samples, dtype="float32"
  )  # type: ignore

  if (
    not isinstance(audio, np.ndarray)
    and audio.dtype != np.float32
    and audio.ndim not in (1, 2)
  ):
    raise Exception("Invalid audio data read from file using soundfile.")

  return convert_to_mono(audio)


def convert_to_mono(
  audio_data: npt.NDArray,
) -> npt.NDArray[np.float32]:
  if audio_data.ndim == 2:
    n_channels = audio_data.shape[1]
    if not n_channels > 1:
      raise Exception("Invalid audio data.")
    result: npt.NDArray[np.float32] = np.mean(audio_data, axis=1, dtype=np.float32)
    return result
  else:
    assert audio_data.ndim == 1
    return audio_data
