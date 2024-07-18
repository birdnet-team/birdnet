
import datetime
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from typing import OrderedDict
from typing import OrderedDict as ODType
from typing import Set, Tuple

import librosa
import numpy as np
import soundfile as sf
from ordered_set import OrderedSet
from scipy.signal import butter, firwin, kaiserord, lfilter

from birdnet.models.v2_4 import SIG_FMAX, SIG_FMIN, MetaDataModelV2p4, ModelV2p4
from birdnet.types import Species, SpeciesList

# Supported file types
# ALLOWED_FILETYPES: Set[str] = {"wav", "flac", "mp3", "ogg", "m4a", "wma", "aiff", "aif"}

SAMPLE_RATE = 48_000
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_species_from_file(file_path: Path, *, encoding: str = "utf8") -> SpeciesList:
  species = SpeciesList(file_path.read_text(encoding).splitlines())
  return species


def get_species_from_location(latitude: float, longitude: float, week: int, *, model_version: str = "2.4", location_filter_threshold: float = 0.03, language: str = "en") -> SpeciesList:
  """Predict a species list.

  Uses the model to predict the species list for the given coordinates and filters by threshold.

  Args:
      lat: The latitude.
      lon: The longitude.
      week: The week of the year [1-48]. Use -1 for year-round.
      threshold: Only values above or equal to threshold will be shown.
      sort: If the species list should be sorted.

  Returns:
      A list of all eligible species.
  """
  if not 0.01 <= location_filter_threshold <= 0.99:
    raise ValueError("location_filter_threshold")

  if model_version == "2.4":
    model = MetaDataModelV2p4()
  else:
    raise ValueError("model_version")

  # Extract species from model
  l_filter = model.predict(latitude, longitude, week)

  # Apply threshold
  l_filter = np.where(l_filter >= location_filter_threshold, l_filter, 0)

  # Zip with labels
  species = model.get_species_by_language(language)
  l_filter = list(zip(l_filter, species))

  # Sort by filter value
  l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

  # Make species list
  species = SpeciesList(
    p[1]
    for p in l_filter
    if p[0] >= location_filter_threshold
  )

  return species


@dataclass()
class AnalysisResult():
  file_path: Path
  model_version: str
  language: str
  sig_length: float
  sig_overlap: float
  sig_minlen: float
  file_splitting_duration: int
  use_bandpass: bool
  bandpass_fmin: Optional[int]
  bandpass_fmax: Optional[int]
  applied_sigmoid: bool
  sigmoid_sensitivity: Optional[float]
  min_confidence: float
  batch_size: int
  available_species: SpeciesList
  filtered_species: Optional[SpeciesList]
  start_time: datetime.datetime
  end_time: datetime.datetime
  duration_seconds: float
  file_duration_seconds: float
  predictions: ODType[Tuple[float, float], ODType[Species, float]]
  sample_rate: int
  model_fmin: int
  model_fmax: int


def analyze_file(
    file_path: Path,
    *,
    model_version: str = "2.4",
    language: str = "en",
    sig_length: float = 3.0,
    sig_overlap: float = 0.0,
    sig_minlen: float = 1.0,
    file_splitting_duration: int = 600,
    use_bandpass: bool = True,
    bandpass_fmin: Optional[int] = 0,
    bandpass_fmax: Optional[int] = 15000,
    # cpu_threads: int = 8,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: Optional[float] = 1.0,
    min_confidence: float = 0.1,
    batch_size: int = 1,
    filter_species: Optional[SpeciesList] = None,
  ) -> Dict:
  """
  sig_minlen: Define minimum length of audio chunk for prediction; chunks shorter than 3 seconds will be padded with zeros
  """

  if not 0.01 <= min_confidence <= 0.99:
    raise ValueError("min_confidence")

  # TODO check if correct
  if not 0.5 <= sigmoid_sensitivity <= 1.5:
    raise ValueError("sigmoid_sensitivity")

  if not 0.0 <= sig_overlap < sig_length:
    raise ValueError("sig_overlap")

  # assert os.cpu_count() is not None
  # if not 1 <= cpu_threads <= os.cpu_count():
  #   raise ValueError("cpu_threads")

  start_time = datetime.datetime.now()
  offset = 0
  start, end = 0, sig_length

  file_duration_seconds = librosa.get_duration(filename=str(file_path.absolute()), sr=SAMPLE_RATE)

  if model_version == "2.4":
    model = ModelV2p4()
  else:
    raise ValueError("model_version")

  species = model.get_species_by_language(language)
  use_species_filter = filter_species is not None
  if use_species_filter:
    species_filter_contains_unknown_species = not filter_species.issubset(species)
    if species_filter_contains_unknown_species:
      raise ValueError("filter_species")
    if len(filter_species) == 0:
      raise ValueError("filter_species")

  predictions = {}

  # Process each chunk
  while offset < file_duration_seconds:
    # will resample to SAMPLE_RATE
    audio_signal, _ = librosa.load(
      file_path,
      sr=SAMPLE_RATE,
      offset=offset,
      duration=file_splitting_duration,
      mono=True,
      res_type="kaiser_fast",
    )

    # Bandpass filter
    if use_bandpass:
      if bandpass_fmin is None:
        raise ValueError("bandpass_fmin")
      if bandpass_fmax is None:
        raise ValueError("bandpass_fmax")

      audio_signal = bandpass_signal(audio_signal, SAMPLE_RATE,
                                     bandpass_fmin, bandpass_fmax, SIG_FMIN, SIG_FMAX)

    samples = []
    timestamps = []

    chunks = split_signal(audio_signal, SAMPLE_RATE, sig_length, sig_overlap, sig_minlen)

    for chunk_index, chunk in enumerate(chunks):
      # Add to batch
      samples.append(chunk)
      timestamps.append([start, end])

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
          prediction,
          sensitivity=-sigmoid_sensitivity,
        )

      # Add to results
      for i in range(len(samples)):
        # Get timestamp
        s_start, s_end = timestamps[i]

        # Get prediction
        pred = prediction[i]

        # Assign scores to labels
        p_labels = zip(species, pred)

        if use_species_filter:
          p_labels = (
            (species, score)
            for species, score in p_labels
            if species in filter_species
          )

        # Sort by score
        preds = OrderedDict(sorted(p_labels, key=operator.itemgetter(1), reverse=True))

        assert (s_start, s_end) not in predictions
        predictions[(s_start, s_end)] = preds

      # Clear batch
      samples = []
      timestamps = []

      # Advance start and end
      start += sig_length - sig_overlap
      end = start + sig_length

    offset = offset + file_splitting_duration

  end_time = datetime.datetime.now()

  result = AnalysisResult(
    file_path=file_path,
    file_duration_seconds=file_duration_seconds,
    use_bandpass=use_bandpass,
    bandpass_fmax=None,
    bandpass_fmin=None,
    applied_sigmoid=apply_sigmoid,
    sigmoid_sensitivity=None,
    available_species=model.get_species_by_language(language),
    filtered_species=filter_species,
    language=language,
    batch_size=batch_size,
    file_splitting_duration=file_splitting_duration,
    min_confidence=min_confidence,
    model_version=model_version,
    sig_length=sig_length,
    sig_minlen=sig_minlen,
    sig_overlap=sig_overlap,
    start_time=start_time,
    end_time=end_time,
    duration_seconds=(end_time - start_time).total_seconds(),
    sample_rate=SAMPLE_RATE,
    model_fmin=SIG_FMIN,
    model_fmax=SIG_FMAX,
    predictions=predictions,
  )

  if use_bandpass:
    result.bandpass_fmax = bandpass_fmax
    result.bandpass_fmin = bandpass_fmin

  if apply_sigmoid:
    result.sigmoid_sensitivity = sigmoid_sensitivity

  return result


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
