import multiprocessing as mp
import os
from collections.abc import Generator, Iterable
from itertools import count, islice
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm

from birdnet.types import Species, TimeInterval
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  get_chunks_with_overlap,
  itertools_batched,
  resample_array,
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


class Producer:
  def __init__(
    self,
    files: List[Path],
    batch_size: int,
    n_slots: int,
    n_jobs: int,
    queue: Queue,
    sem_free: Semaphore,  # counts free slots
    sem_fill: Semaphore,  # counts filled slots
    chunk_duration_s: float = 3.0,
    overlap_duration_s: float = 0.0,
    target_sample_rate: int = 48000,
    use_bandpass: bool = False,
    bandpass_fmin: Optional[int] = None,
    bandpass_fmax: Optional[int] = None,
    fmin: Optional[int] = None,
    fmax: Optional[int] = None,
  ):
    self.chunk_duration_s = chunk_duration_s
    self.overlap_duration_s = overlap_duration_s
    self.target_sample_rate = target_sample_rate
    self._batch_size = batch_size
    self._n_jobs = n_jobs
    self._n_slots = n_slots
    self._sem_free = sem_free
    self._sem_fill = sem_fill
    self._write_ptr = 0
    self._queue = queue
    self._files = files
    self.use_bandpass = use_bandpass

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

    # attach to existing shared memory buffers
    # NOTE: these handlers must be created that GC does not delete the shared memory access
    self._shm_file_indices = SharedMemory(name="bnet_ring_file_indices", create=False)
    self._shm_chunk_indices = SharedMemory(name="bnet_ring_chunk_indices", create=False)
    self._shm_audio_samples = SharedMemory(name="bnet_ring_audio_samples", create=False)

    self._ring_file_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, self._shm_file_indices.buf
    )
    self._ring_chunk_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, self._shm_chunk_indices.buf
    )
    self._ring_audio_samples = np.ndarray(
      (n_slots, batch_size, self.chunk_duration_samples),
      np.float32,
      self._shm_audio_samples.buf,
    )

  def get_chunks_from_files(
    self,
  ) -> Generator[tuple[int, int, npt.NDArray[np.float32]], None, None]:
    for file_index, path in enumerate(self._files):
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

      if self.use_bandpass:
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
    assert self._queue.empty()

    buffer_input = self.get_chunks_from_files()
    for batch in itertools_batched(buffer_input, self._batch_size):
      file_indices, chunk_indices, audio_samples = zip(*batch)
      self._flush_batch(file_indices, chunk_indices, audio_samples)

    # send poison pills
    for _ in range(self._n_jobs):
      self._queue.put(None)

  def _flush_batch(self, file_indices, chunk_indices, audio_samples) -> None:
    self._sem_free.acquire()
    slot = self._write_ptr % self._n_slots
    self._write_ptr += 1
    current_batch_size = len(audio_samples)
    assert len(file_indices) == current_batch_size
    assert len(chunk_indices) == current_batch_size
    assert 0 <= slot < self._n_slots
    assert current_batch_size <= self._batch_size
    self._ring_file_indices[slot, :current_batch_size] = np.asarray(
      file_indices, np.uint32
    )
    self._ring_chunk_indices[slot, :current_batch_size] = np.asarray(
      chunk_indices, np.uint32
    )
    self._ring_audio_samples[slot, :current_batch_size] = np.asarray(
      np.stack(audio_samples, 0), np.float32
    )
    self._queue.put((slot, current_batch_size))
    self._sem_fill.release()


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
def shm_ring(name: str, size: int):
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
    shm_ring(
      "bnet_ring_file_indices",
      n_slots * batch_size * np.dtype(np.uint32).itemsize,
    ) as shm_file_indices,
    shm_ring(
      "bnet_ring_chunk_indices", n_slots * batch_size * np.dtype(np.uint32).itemsize
    ) as shm_chunk_indices,
    shm_ring(
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
