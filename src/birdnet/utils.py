from collections.abc import Generator, Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet
from tqdm import tqdm


def get_species_from_file(
  species_file: Path, /, *, encoding: str = "utf8"
) -> OrderedSet[str]:
  species = OrderedSet(species_file.read_text(encoding).strip().splitlines())
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
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  # Lowpass
  elif fmin == new_fmin and fmax < new_fmax:
    high = fmax / nyquist
    b, a = butter(nth_order, high, btype="low")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  # Bandpass
  elif fmin > new_fmin and fmax < new_fmax:
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(nth_order, [low, high], btype="band")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  sig_f32 = audio_signal.astype(np.float32)
  return sig_f32


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


def flat_sigmoid_logaddexp_fast(
  x: npt.NDArray, sensitivity: float, clip_val: float = 15.0
) -> npt.NDArray:
  y = sensitivity * np.clip(x, -clip_val, clip_val)

  positive_mask = y >= 0
  abs_y = np.abs(y)
  exp_neg_abs = np.exp(-abs_y, dtype=x.dtype)

  one_plus_exp = 1.0 + exp_neg_abs

  return np.where(positive_mask, exp_neg_abs / one_plus_exp, 1.0 / one_plus_exp)


def download_file_tqdm(
  url: str,
  file_path: Path,
  *,
  download_size: int | None = None,
  description: str | None = None,
) -> int:
  assert file_path.parent.is_dir()
  import requests

  response = requests.get(url, stream=True, timeout=30)
  total_size = int(response.headers.get("content-length", 0))
  if download_size is not None:
    total_size = download_size

  block_size = 1024
  with (
    tqdm(total=total_size, unit="iB", unit_scale=True, desc=description) as tqdm_bar,
    open(file_path, "wb") as file,
  ):
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
