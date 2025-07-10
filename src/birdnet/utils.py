import os
from collections.abc import Generator, Iterable
from itertools import count, islice
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import soundfile as sf
from ordered_set import OrderedSet
from tqdm import tqdm


def get_species_from_file(
  species_file: Path, /, *, encoding: str = "utf8"
) -> OrderedSet[str]:
  species = OrderedSet(species_file.read_text(encoding).splitlines())
  return species


def bandpass_signal(
  audio_signal: npt.NDArray[np.float32],
  rate: int,
  fmin: int,
  fmax: int,
  new_fmin: int,
  new_fmax: int,
) -> npt.NDArray[np.float32]:
  assert rate > 0
  assert fmin >= 0
  assert fmin < fmax
  assert new_fmin >= 0
  assert new_fmin < new_fmax

  from scipy.signal import butter, lfilter

  nth_order = 5
  nyquist = rate // 2

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


def segment_signal(
  audio_signal: npt.NDArray[np.float32],
  rate: int,
  segment_size: float,
  segment_overlap: float,
  min_segment_size: float,
) -> Generator[Tuple[float, float, npt.NDArray[np.float32]], None, None]:
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
  assert min_segment_size > 0
  assert segment_overlap >= 0
  assert segment_overlap < segment_size

  # Number of frames per segment, per step and per minimum signal
  segment_frame_count = round(rate * segment_size)
  segment_step_frame_count = round(rate * (segment_size - segment_overlap))
  min_segment_frame_count = round(rate * min_segment_size)

  # Start of last segment
  last_segment_position = (
    round(
      (audio_signal.size - segment_frame_count + segment_step_frame_count - 1)
      / segment_step_frame_count
    )
    * segment_step_frame_count
  )
  # Make sure at least one segment is returned
  if last_segment_position < 0:
    last_segment_position = 0
  # Omit last segment if minimum signal duration is underrun
  elif audio_signal.size - last_segment_position < min_segment_frame_count:
    last_segment_position = last_segment_position - segment_step_frame_count

  # Append empty signal of segment duration, so the last split has the desired length in any case
  # TODO maybe add noise instead of empty signal
  noise = np.zeros(shape=segment_frame_count, dtype=audio_signal.dtype)

  data = np.concatenate((audio_signal, noise))
  start: float = 0.0
  end: float = segment_size

  # Split signal with overlap
  for i in range(0, 1 + last_segment_position, segment_step_frame_count):
    segment = data[i : i + segment_frame_count]

    yield start, end, segment

    # Advance start and end
    start += segment_size - segment_overlap
    end = start + segment_size


def fillup_with_silence(
  audio_segment: npt.NDArray[np.float32], target_length: int
) -> npt.NDArray[np.float32]:
  current_length = len(audio_segment)
  assert current_length <= target_length

  if current_length == target_length:
    return audio_segment

  silence_length = target_length - current_length
  silence = np.zeros(silence_length, dtype=audio_segment.dtype)
  filled_segment = np.concatenate((audio_segment, silence))

  return filled_segment


def flat_sigmoid(
  x: npt.NDArray[np.float32], sensitivity: float
) -> npt.NDArray[np.float32]:
  result: npt.NDArray[np.float32] = 1.0 / (
    1.0 + np.exp(sensitivity * np.clip(x, -15, 15))
  )
  return result


def sigmoid_inverse(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
  return np.log(x / (1 - x))


def get_app_data_path() -> Path:
  """Returns the appropriate application data path based on the operating system."""
  if os.name == "nt":  # Windows
    app_data_path = os.getenv("APPDATA")
    assert app_data_path is not None
  elif os.name == "posix":
    if os.uname().sysname == "Darwin":  # Mac OS X
      app_data_path = os.path.expanduser("~/Library/Application Support")
    else:  # Linux
      app_data_path = os.path.expanduser("~/.local/share")
  else:
    raise OSError("Unsupported operating system")

  result = Path(app_data_path)
  return result


def get_birdnet_app_data_folder() -> Path:
  app_data = get_app_data_path()
  result = app_data / "birdnet"
  return result


def download_file(url: str, file_path: Path) -> None:
  assert file_path.parent.is_dir()
  import requests

  response = requests.get(url, timeout=30)
  if response.status_code == 200:
    with open(file_path, "wb") as file:
      file.write(response.content)
  else:
    raise ValueError(
      f"Failed to download the file. Status code: {response.status_code}"
    )


def download_file_tqdm(
  url: str,
  file_path: Path,
  *,
  download_size: Optional[int] = None,
  description: Optional[str] = None,
) -> int:
  assert file_path.parent.is_dir()
  import requests

  response = requests.get(url, stream=True, timeout=30)
  total_size = int(response.headers.get("content-length", 0))
  if download_size is not None:
    total_size = download_size

  block_size = 1024
  with tqdm(total=total_size, unit="iB", unit_scale=True, desc=description) as tqdm_bar:
    with open(file_path, "wb") as file:
      for data in response.iter_content(block_size):
        tqdm_bar.update(len(data))
        file.write(data)

  if response.status_code != 200 or (total_size not in (0, tqdm_bar.n)):
    raise ValueError(
      f"Failed to download the file. Status code: {response.status_code}"
    )
  return total_size


def itertools_batched(iterable: Iterable, n: int) -> Generator[Any, None, None]:
  # https://docs.python.org/3.12/library/itertools.html#itertools.batched
  # batched('ABCDEFG', 3) → ABC DEF G
  if n < 1:
    raise ValueError("n must be at least one")
  iterator = iter(iterable)
  while batch := tuple(islice(iterator, n)):
    yield batch


def get_segments_with_overlap(
  total_duration_s: Union[int, float],
  segment_duration_s: Union[int, float],
  overlap_duration_s: Union[int, float],
) -> Generator[Tuple[float, float], None, None]:
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
  for start in count(0.0, step_duration):
    assert start < total_duration_s
    if (end := start + segment_duration_s) < total_duration_s:
      yield start, end
    else:
      yield start, total_duration_s
      break


def iter_segments_with_overlap(
  segment_duration_s: Union[int, float],
  overlap_duration_s: Union[int, float],
  /,
  *,
  start: Union[int, float] = 0.0,
) -> Generator[Tuple[float, float], None, None]:
  assert segment_duration_s > 0
  assert 0 <= overlap_duration_s < segment_duration_s

  if not isinstance(overlap_duration_s, float):
    overlap_duration_s = float(overlap_duration_s)
  if not isinstance(segment_duration_s, float):
    segment_duration_s = float(segment_duration_s)
  if not isinstance(start, float):
    start = float(start)

  step_duration = segment_duration_s - overlap_duration_s

  for s in count(start, step_duration):
    end = s + segment_duration_s
    yield s, end
