import os
from itertools import islice
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Tuple, cast

import librosa
import numpy as np
import numpy.typing as npt
import requests
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter
from tqdm import tqdm

from birdnet.types import Species


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

  # Append empty signal of chunk duration, so all splits have desired length
  noise = np.zeros(shape=chunk_frame_count, dtype=audio_signal.dtype)
  # TODO maybe add noise

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


def flat_sigmoid(x: npt.NDArray[np.float32], sensitivity: float) -> npt.NDArray[np.float32]:
  result: npt.NDArray[np.float32] = 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))
  return result


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


def load_audio_file_in_parts(audio_file: Path, sample_rate: int, file_splitting_duration: float) -> Generator[npt.NDArray[np.float32], None, None]:
  offset = 0.0
  file_duration_seconds = cast(float, librosa.get_duration(
      path=str(audio_file.absolute()), sr=sample_rate))

  while offset < file_duration_seconds:
    # will resample to sample_rate
    audio_signal, _ = librosa.load(
        audio_file,
        sr=sample_rate,
        offset=offset,
        duration=file_splitting_duration,
        mono=True,
        res_type="kaiser_fast",
    )
    audio_signal = cast(npt.NDArray[np.float32], audio_signal)
    yield audio_signal
    del audio_signal
    offset += file_splitting_duration
