import os
from itertools import count, islice
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm

from birdnet.types import Species, TimeInterval


def get_species_from_file(species_file: Path, /, *, encoding: str = "utf8") -> OrderedSet[Species]:
  species = OrderedSet(species_file.read_text(encoding).splitlines())
  return species


def bandpass_signal(audio_signal: npt.NDArray[np.float32], rate: int, fmin: int, fmax: int, new_fmin: int, new_fmax: int) -> npt.NDArray[np.float32]:
  assert rate > 0
  assert fmin >= 0
  assert fmin < fmax
  assert new_fmin >= 0
  assert new_fmin < new_fmax

  nth_order = 5
  nyquist = 0.5 * rate

  # Highpass
  if fmin > new_fmin and fmax == new_fmax:
    low = fmin / nyquist
    b, a = butter(nth_order, low, btype="high")
    audio_signal = lfilter(b, a, audio_signal)

  # Lowpass
  elif fmin == new_fmin and fmax < new_fmax:
    high = fmax / nyquist
    b, a = butter(nth_order, high, btype="low")
    audio_signal = lfilter(b, a, audio_signal)

  # Bandpass
  elif fmin > new_fmin and fmax < new_fmax:
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(nth_order, [low, high], btype="band")
    audio_signal = lfilter(b, a, audio_signal)

  sig_f32 = audio_signal.astype(np.float32)
  return sig_f32


def chunk_signal(audio_signal: npt.NDArray[np.float32], rate: int, chunk_size: float, chunk_overlap: float, min_chunk_size: float) -> Generator[Tuple[float, float, npt.NDArray[np.float32]], None, None]:
  """Split signal with overlap.

  Args:
      sig: The original signal to be split.
      rate: The sampling rate.
      seconds: The duration of a segment.
      overlap: The overlapping seconds of segments.
      minlen: Minimum length of a split.

  Returns:
      A list of splits.
  """
  assert rate > 0
  assert min_chunk_size > 0
  assert chunk_overlap >= 0
  assert chunk_overlap < chunk_size

  # Number of frames per chunk, per step and per minimum signal
  chunk_frame_count = round(rate * chunk_size)
  chunk_step_frame_count = round(rate * (chunk_size - chunk_overlap))
  min_chunk_frame_count = round(rate * min_chunk_size)

  # Start of last chunk
  last_chunk_position = round((audio_signal.size - chunk_frame_count +
                              chunk_step_frame_count - 1) / chunk_step_frame_count) * chunk_step_frame_count
  # Make sure at least one chunk is returned
  if last_chunk_position < 0:
    last_chunk_position = 0
  # Omit last chunk if minimum signal duration is underrun
  elif audio_signal.size - last_chunk_position < min_chunk_frame_count:
    last_chunk_position = last_chunk_position - chunk_step_frame_count

  # Append empty signal of chunk duration, so the last split has the desired length in any case
  # TODO maybe add noise instead of empty signal
  noise = np.zeros(shape=chunk_frame_count, dtype=audio_signal.dtype)

  data = np.concatenate((audio_signal, noise))
  start: float = 0.0
  end: float = chunk_size

  # Split signal with overlap
  for i in range(0, 1 + last_chunk_position, chunk_step_frame_count):
    chunk = data[i:i + chunk_frame_count]

    yield start, end, chunk

    # Advance start and end
    start += chunk_size - chunk_overlap
    end = start + chunk_size


def fillup_with_silence(audio_chunk: npt.NDArray[np.float32], target_length: int) -> npt.NDArray[np.float32]:
  current_length = len(audio_chunk)
  assert current_length <= target_length

  if current_length == target_length:
    return audio_chunk

  silence_length = target_length - current_length
  silence = np.zeros(silence_length, dtype=audio_chunk.dtype)
  filled_chunk = np.concatenate((audio_chunk, silence))

  return filled_chunk


def flat_sigmoid(x: npt.NDArray[np.float32], sensitivity: float) -> npt.NDArray[np.float32]:
  result: npt.NDArray[np.float32] = 1.0 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))
  return result


def sigmoid_inverse(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
  return np.log(x / (1 - x))


def get_app_data_path() -> Path:
  """Returns the appropriate application data path based on the operating system."""
  if os.name == 'nt':  # Windows
    app_data_path = os.getenv('APPDATA')
    assert app_data_path is not None
  elif os.name == 'posix':
    if os.uname().sysname == 'Darwin':  # Mac OS X
      app_data_path = os.path.expanduser('~/Library/Application Support')
    else:  # Linux
      app_data_path = os.path.expanduser('~/.local/share')
  else:
    raise OSError('Unsupported operating system')

  result = Path(app_data_path)
  return result


def get_birdnet_app_data_folder() -> Path:
  app_data = get_app_data_path()
  result = app_data / "birdnet"
  return result


def download_file(url: str, file_path: Path) -> None:
  assert file_path.parent.is_dir()

  response = requests.get(url, timeout=30)
  if response.status_code == 200:
    with open(file_path, 'wb') as file:
      file.write(response.content)
  else:
    raise ValueError(f"Failed to download the file. Status code: {response.status_code}")


def download_file_tqdm(url: str, file_path: Path, *, download_size: Optional[int] = None, description: Optional[str] = None) -> None:
  assert file_path.parent.is_dir()

  response = requests.get(url, stream=True, timeout=30)
  total_size = int(response.headers.get('content-length', 0))
  if download_size is not None:
    total_size = download_size

  block_size = 1024
  with tqdm(total=total_size, unit='iB', unit_scale=True, desc=description) as tqdm_bar:
    with open(file_path, 'wb') as file:
      for data in response.iter_content(block_size):
        tqdm_bar.update(len(data))
        file.write(data)

  if response.status_code != 200 or (total_size not in (0, tqdm_bar.n)):
    raise ValueError(f"Failed to download the file. Status code: {response.status_code}")


def itertools_batched(iterable: Iterable, n: int) -> Generator[Any, None, None]:
  # https://docs.python.org/3.12/library/itertools.html#itertools.batched
  # batched('ABCDEFG', 3) → ABC DEF G
  if n < 1:
    raise ValueError('n must be at least one')
  iterator = iter(iterable)
  while batch := tuple(islice(iterator, n)):
    yield batch


def get_chunks_with_overlap(total_duration_s: Union[int, float], chunk_duration_s: Union[int, float], overlap_duration_s: Union[int, float]) -> Generator[Tuple[float, float], None, None]:
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


def iter_chunks_with_overlap(chunk_duration_s: Union[int, float], overlap_duration_s: Union[int, float], /, *, start: Union[int, float] = 0.0) -> Generator[Tuple[float, float], None, None]:
  assert chunk_duration_s > 0
  assert 0 <= overlap_duration_s < chunk_duration_s

  if not isinstance(overlap_duration_s, float):
    overlap_duration_s = float(overlap_duration_s)
  if not isinstance(chunk_duration_s, float):
    chunk_duration_s = float(chunk_duration_s)
  if not isinstance(start, float):
    start = float(start)

  step_duration = chunk_duration_s - overlap_duration_s

  for s in count(start, step_duration):
    end = s + chunk_duration_s
    yield s, end


def resample_array(x: npt.NDArray, sample_rate: int, target_sample_rate: int) -> npt.NDArray:
  assert len(x.shape) == 1
  assert 0 < sample_rate
  assert 0 < target_sample_rate

  if sample_rate == target_sample_rate:
    return x

  target_sample_count = round(len(x) / sample_rate * target_sample_rate)
  x_resampled: npt.NDArray = resample(x, target_sample_count)
  assert x_resampled.dtype == x.dtype
  return x_resampled


def load_audio_in_chunks_with_overlap(audio_path: Path, /, *, chunk_duration_s: float = 3, overlap_duration_s: float = 0, target_sample_rate: int = 48000) -> Generator[Tuple[float, float, npt.NDArray[np.float32]], None, None]:
  assert audio_path.is_file()

  sf_info = sf.info(audio_path)
  sample_rate = sf_info.samplerate

  timestamps = get_chunks_with_overlap(
    float(sf_info.duration),
    float(chunk_duration_s),
    float(overlap_duration_s),
  )

  for start, end in timestamps:
    start_samples = round(start * sample_rate)
    end_samples = round(end * sample_rate)
    audio, _ = sf.read(audio_path, start=start_samples, stop=end_samples, dtype=np.float32)
    audio = resample_array(audio, sample_rate, target_sample_rate)
    yield start, end, audio


def iter_audio_in_chunks_with_overlap(audio_path: Path, /, *, chunk_duration_s: float = 3, overlap_duration_s: float = 0, target_sample_rate: int = 48000) -> Generator[Tuple[TimeInterval, npt.NDArray[np.float32]], None, None]:
  # same method as above
  assert audio_path.is_file()

  sf_info = sf.info(audio_path)
  sample_rate = sf_info.samplerate
  file_duration = float(sf_info.duration)

  timestamps = iter_chunks_with_overlap(
    float(chunk_duration_s),
    float(overlap_duration_s),
    start=0.0,
  )

  for start, end in timestamps:
    assert start < file_duration
    start_samples = round(start * sample_rate)
    end = min(end, file_duration)
    end_samples = round(end * sample_rate)
    audio, _ = sf.read(audio_path, start=start_samples, stop=end_samples, dtype=np.float32)
    audio = resample_array(audio, sample_rate, target_sample_rate)
    yield (start, end), audio
    was_last_chunk = end == file_duration
    if was_last_chunk:
      return
