
import datetime
import operator
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, firwin, kaiserord, lfilter

from birdnet.models.v2_4 import SIG_FMAX, SIG_FMIN, Model_v2_4

# Supported file types
ALLOWED_FILETYPES: Set[str] = {"wav", "flac", "mp3", "ogg", "m4a", "wma", "aiff", "aif"}

SAMPLE_RATE = 48_000


def getAudioFileLength(path: Path) -> int:
  # Open file with librosa (uses ffmpeg or libav)
  return round(librosa.get_duration(filename=path, sr=SAMPLE_RATE))


def analyze_file(
    file_path: Path,
    *,
    lat: Optional[float] = None,
    long: Optional[float] = None,
    week: Optional[int] = None,
    location_filter_threshold: float = 0.03,
    custom_species_path: Optional[Path] = None,
    sig_length: float = 3.0,
    sig_overlap: float = 0.0,
    sig_minlen: float = 1.0,
    bandpass: bool = True,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15000,
    cpu_threads: int = 8,    
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float = 1.0,
    min_confidence: float = 0.1,
    batch_size: int = 1,
    file_splitting_duration: int = 600,
  ) -> Dict:
  """
  sig_minlen: Define minimum length of audio chunk for prediction; chunks shorter than 3 seconds will be padded with zeros
  """

  if not 0.01 <= location_filter_threshold <= 0.99:
    raise ValueError("location_filter_threshold")

  if not 0.01 <= min_confidence <= 0.99:
    raise ValueError("min_confidence")

  # TODO check if correct
  if not 0.5 <= sigmoid_sensitivity <= 1.5:
    raise ValueError("sigmoid_sensitivity")

  if not 0.0 <= sig_overlap < sig_length:
    raise ValueError("sig_overlap")

  assert os.cpu_count() is not None
  if not 1 <= cpu_threads <= os.cpu_count():
    raise ValueError("cpu_threads")

  # TODO species list prediction
  custom_species = None
  if custom_species_path:
    custom_species = custom_species_path.read_text("utf8").splitlines()

  start_time = datetime.datetime.now()
  offset = 0
  start, end = 0, sig_length

  fileLengthSeconds = librosa.get_duration(filename=str(file_path.absolute()), sr=SAMPLE_RATE)

  model = Model_v2_4()
  results = {}

  # Process each chunk
  while offset < fileLengthSeconds:
    # will resample to SAMPLE_RATE
    sig, _ = librosa.load(
      file_path,
      sr=SAMPLE_RATE,
      offset=offset,
      duration=file_splitting_duration,
      mono=True,
      res_type="kaiser_fast",
    )

    # Bandpass filter
    if bandpass:
      sig = bandpass_signal(sig, SAMPLE_RATE, bandpass_fmin, bandpass_fmax, SIG_FMIN, SIG_FMAX)
      # sig = bandpassKaiserFIR(sig, rate, fmin, fmax)

    samples = []
    timestamps = []

    chunks = splitSignal(sig, SAMPLE_RATE, sig_length, sig_overlap, sig_minlen)

    for chunk_index, chunk in enumerate(chunks):
      # Add to batch
      samples.append(chunk)
      timestamps.append([start, end])

      # Advance start and end
      start += sig_length - sig_overlap
      end = start + sig_length

      # Check if batch is full or last chunk
      if len(samples) < batch_size and chunk_index < len(chunks) - 1:
        continue

      # Predict
      # Prepare sample and pass through birdnet.model
      data = np.array(samples, dtype="float32")
      prediction = model.predict(data)

      # Logits or sigmoid activations?
      if apply_sigmoid:
        prediction = flat_sigmoid(
          np.array(prediction), sensitivity=-sigmoid_sensitivity
        )

      # Add to results
      for i in range(len(samples)):
        # Get timestamp
        s_start, s_end = timestamps[i]

        # Get prediction
        pred = prediction[i]

        # Assign scores to labels
        p_labels = zip(model.labels, pred)

        # Sort by score
        p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)

        # Store top 5 results and advance indices
        results[str(s_start) + "-" + str(s_end)] = p_sorted

      # Clear batch
      samples = []
      timestamps = []
    offset = offset + file_splitting_duration

  return results


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


def splitSignal(sig, rate, seconds, overlap, minlen) -> List:
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


def flat_sigmoid(x, sensitivity=-1):
  return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))
