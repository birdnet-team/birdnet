import os
from collections.abc import Generator, Iterable
from itertools import count, islice
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
from birdnet.utils import get_chunks_with_overlap, resample_array


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


import multiprocessing as mp
from multiprocessing.synchronize import Event


class Producer:
  def __init__(
    self,
    files: List[Path],
    chunk_duration_s: float = 3.0,
    overlap_duration_s: float = 0.0,
    target_sample_rate: int = 48000,
    queue_size: int = 16,
  ):
    self.chunk_duration_s = chunk_duration_s
    self.overlap_duration_s = overlap_duration_s
    self.target_sample_rate = target_sample_rate
    self._queue = mp.Queue(maxsize=queue_size)
    self._files = files
    self.reading_finished = mp.Event()

  @property
  def queue(self) -> mp.Queue:
    """
    Returns the queue used for processing audio files.
    """
    return self._queue

  def fill_queue(
    self,
  ) -> None:
    assert not self.reading_finished.is_set()
    assert self._queue.empty()
    # self.reading_finished.clear()
    # self._queue.close()  # Close the queue before filling it
    for file_idx, path in enumerate(self._files):
      chunks = load_audio_in_chunks_with_overlap(
        path,
        chunk_duration_s=self.chunk_duration_s,
        overlap_duration_s=self.overlap_duration_s,
        target_sample_rate=self.target_sample_rate,
      )

      for chunk_idx, batch in chunks:
        self._queue.put((file_idx, chunk_idx, batch))
    self.reading_finished.set()  # Signal that processing is done

  def load_audio_in_chunks_with_overlap(self, audio_path: Path) -> Generator:
    return load_audio_in_chunks_with_overlap(
      audio_path,
      chunk_duration_s=self.chunk_duration_s,
      overlap_duration_s=self.overlap_duration_s,
      target_sample_rate=self.target_sample_rate,
    )


def load_audio_in_chunks_with_overlap(
  audio_path: Path,
  /,
  *,
  chunk_duration_s: float = 3,
  overlap_duration_s: float = 0,
  # read_duration_s: Optional[float] = None,
  target_sample_rate: int = 48000,
) -> Generator[tuple[int, npt.NDArray[np.float32]], None, None]:
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

  for idx, (start, end) in enumerate(timestamps):
    start_samples = round(start * sample_rate)
    end_samples = round(end * sample_rate)
    audio, _ = sf.read(
      audio_path, start=start_samples, stop=end_samples, dtype=np.float32
    )
    audio = resample_array(audio, sample_rate, target_sample_rate)
    yield idx, audio


if __name__ == "__main__":
  # Example usage
  producer = Producer(files=[Path("example/soundscape.wav")])
  producer.fill_queue()

  while not producer.queue.empty():
    file_idx, chunk_idx, batch = producer.queue.get()
    print(
      f"File Index: {file_idx}, Chunk Index: {chunk_idx}, Batch Shape: {batch.shape}"
    )
