from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import butter, lfilter

from birdnet.types import SpeciesList


def get_species_from_file(file_path: Path, *, encoding: str = "utf8") -> SpeciesList:
  species = SpeciesList(file_path.read_text(encoding).splitlines())
  return species


def bandpass_signal(sig, rate: int, fmin: int, fmax: int, new_fmin: int, new_fmax: int):
  nth_order = 5
  nyquist = 0.5 * rate

  # Highpass
  if fmin > new_fmin and fmax == new_fmax:
    low = fmin / nyquist
    b, a = butter(nth_order, low, btype="high")
    sig = lfilter(b, a, sig)

  # Lowpass
  elif fmin == new_fmin and fmax < new_fmax:
    high = fmax / nyquist
    b, a = butter(nth_order, high, btype="low")
    sig = lfilter(b, a, sig)

  # Bandpass
  elif fmin > new_fmin and fmax < new_fmax:
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(nth_order, [low, high], btype="band")
    sig = lfilter(b, a, sig)

  sig_f32 = sig.astype("float32")
  return sig_f32


def split_signal(sig, rate: int, seconds: float, overlap: float, minlen: float) -> List:
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
  assert overlap < seconds

  # Number of frames per chunk, per step and per minimum signal
  chunksize = round(rate * seconds)
  stepsize = round(rate * (seconds - overlap))
  minsize = round(rate * minlen)

  # Start of last chunk
  lastchunkpos = round((sig.size - chunksize + stepsize - 1) / stepsize) * stepsize
  # Make sure at least one chunk is returned
  if lastchunkpos < 0:
    lastchunkpos = 0
  # Omit last chunk if minimum signal duration is underrun
  elif sig.size - lastchunkpos < minsize:
    lastchunkpos = lastchunkpos - stepsize

  # Append empty signal of chunk duration, so all splits have desired length
  noise = np.zeros(shape=chunksize, dtype=sig.dtype)
  # TODO maybe add noise

  data = np.concatenate((sig, noise))

  # Split signal with overlap
  sig_splits = []
  for i in range(0, 1 + lastchunkpos, stepsize):
    sig_splits.append(data[i:i + chunksize])

  return sig_splits


def flat_sigmoid(x, sensitivity: int):
  return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))

import os

import requests


def get_app_data_path() -> Path:
  """Returns the appropriate application data path based on the operating system."""
  if os.name == 'nt':  # Windows
    app_data_path = os.getenv('APPDATA')
  elif os.name == 'posix':
    if os.uname().sysname == 'Darwin':  # Mac OS X
      app_data_path = os.path.expanduser('~/Library/Application Support')
    else:  # Linux
      app_data_path = os.path.expanduser('~/.local/share')
  else:
    raise OSError('Unsupported operating system')
  
  return Path(app_data_path)

def get_birdnet_app_data_folder() -> Path:
  app_data = get_app_data_path()
  result = app_data / "birdnet"
  return  result

def download_file(url: str, file_path: Path):
  assert file_path.parent.is_dir()
  
  response = requests.get(url, timeout=30)
  if response.status_code == 200:
    with open(file_path, 'wb') as file:
      file.write(response.content)
  else:
    raise ValueError(f"Failed to download the file. Status code: {response.status_code}")

import requests
from tqdm import tqdm


def download_file_tqdm(url: str, file_path: Path) -> None:
  assert file_path.parent.is_dir()
  
  response = requests.get(url, stream=True, timeout=30)
  total_size = int(response.headers.get('content-length', 0))
  block_size = 1024
  with tqdm(total=total_size, unit='iB', unit_scale=True) as tqdm_bar:
    with open(file_path, 'wb') as file:
      for data in response.iter_content(block_size):
        tqdm_bar.update(len(data))
        file.write(data)

  if response.status_code != 200 or (total_size not in (0, tqdm_bar.n)):
    raise ValueError(f"Failed to download the file. Status code: {response.status_code}")
  