import datetime
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, OrderedDict

import librosa
import numpy as np
import numpy.typing as npt
import tflite_runtime.interpreter as tflite

from birdnet.models.model_base import AnalysisResultBase, ModelBase
from birdnet.types import SpeciesList
from birdnet.utils import (bandpass_signal, download_file_tqdm, flat_sigmoid,
                           get_birdnet_app_data_folder, split_signal)

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODEL_VERSION = "v2.4"


@dataclass()
class AnalysisResultV2p4(AnalysisResultBase):
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
  sample_rate: int
  model_fmin: int
  model_fmax: int
  
import zipfile

AVAILABLE_LANGUAGES = {
"sv", "da", "hu", "th", "pt", "fr", "cs", "af", "en_uk", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "en_us", "no", "sk", "ar", "fi", "ro", "nl", "zh"
}

class Downloader():
  def __init__(self) -> None:
    birdnet_app_data = get_birdnet_app_data_folder()
    self._version_path = birdnet_app_data / "models" / MODEL_VERSION
    self._audio_model_path = self._version_path / "audio-model.tflite"
    self._meta_model_path = self._version_path / "meta-model.tflite"
    self._lang_path = self._version_path / "labels"
  
  @property
  def audio_model_path(self) -> Path:
    return self._audio_model_path  
  
  @property
  def meta_model_path(self) -> Path:
    return self._meta_model_path
  
  def get_language_path(self, language: str) -> Path:
    return self._lang_path / f"{language}.txt"
  
  def _check_model_files_exist(self) -> bool:
    model_is_downloaded = True
    model_is_downloaded &= self._audio_model_path.is_file()
    model_is_downloaded &= self._meta_model_path.is_file()
    model_is_downloaded &= self._lang_path.is_dir()
    for lang in AVAILABLE_LANGUAGES:
      model_is_downloaded &= self.get_language_path(lang).is_file()
    return model_is_downloaded
  
  def _download_model_files(self):
    dl_path = "https://tuc.cloud/index.php/s/45KmTcpHH8iDDA2/download/BirdNET_v2.4.zip"
    self._version_path.mkdir(parents=True, exist_ok=True)
    
    zip_download_path = self._version_path / "download.zip"
    
    print("Downloading model ...")
    download_file_tqdm(dl_path, zip_download_path)

    print("Extracting model ...")
    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
      zip_ref.extractall(self._version_path)
    print("Done.")
    
    os.remove(zip_download_path)
    
    
  
  def ensure_model_is_available(self) -> None:
    if not self._check_model_files_exist():
      self._download_model_files()
      assert self._check_model_files_exist()
    

class ModelV2p4(ModelBase):
  def __init__(self, tflite_threads: int = 1, language: str="en_us") -> None:
    super().__init__()
    if language not in AVAILABLE_LANGUAGES:
      raise ValueError(f"Language '{language}' is not available! Choose from: {','.join(sorted(AVAILABLE_LANGUAGES))}")
    
    self.language = language
    downloader = Downloader()
    downloader.ensure_model_is_available()

    # Frequency range. This is model specific and should not be changed.
    self.sig_fmin: int = 0
    self.sig_fmax: int = 15_000
    self.sample_rate = 48_000
    
    self._species_list = SpeciesList(downloader.get_language_path(language).read_text("utf8").splitlines())
    # [line.split(",")[1] for line in labels]

    # Load TFLite model and allocate tensors.
    self.audio_interpreter = tflite.Interpreter(str(downloader.audio_model_path.absolute()), num_threads=tflite_threads)
    # Get input tensor index
    input_details = self.audio_interpreter.get_input_details()
    self.audio_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self.audio_interpreter.get_output_details()
    self.audio_output_layer_index = output_details[0]["index"]
    self.audio_interpreter.allocate_tensors()
    
    # Load TFLite model and allocate tensors.
    self.meta_interpreter = tflite.Interpreter(str(downloader.meta_model_path.absolute()), num_threads=tflite_threads)
    # Get input tensor index
    input_details = self.meta_interpreter.get_input_details()
    self.meta_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self.meta_interpreter.get_output_details()
    self.meta_output_layer_index = output_details[0]["index"]
    self.meta_interpreter.allocate_tensors()

  @property
  def species(self):
    return self._species_list

  def __predict_audio(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert sample.dtype == np.float32
    # Same method as embeddings

    self.audio_interpreter.resize_tensor_input(self.audio_input_layer_index, sample.shape)
    self.audio_interpreter.allocate_tensors()

    # Make a prediction (Audio only for now)
    self.audio_interpreter.set_tensor(self.audio_input_layer_index, sample)
    self.audio_interpreter.invoke()
    prediction = self.audio_interpreter.get_tensor(self.audio_output_layer_index)

    return prediction

  def __predict_meta(self, lat, lon, week):
    # Prepare mdata as sample
    sample = np.expand_dims(np.array([lat, lon, week], dtype="float32"), 0)

    # Run inference
    self.meta_interpreter.set_tensor(self.meta_input_layer_index, sample)
    self.meta_interpreter.invoke()

    prediction = self.meta_interpreter.get_tensor(self.meta_output_layer_index)[0]
    return prediction

  def get_species_from_location(self, latitude: float, longitude: float, week: int, *, location_filter_threshold: float = 0.03) -> SpeciesList:
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

    # Extract species from model
    l_filter = self.__predict_meta(latitude, longitude, week)

    # Apply threshold
    l_filter = np.where(l_filter >= location_filter_threshold, l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, self._species_list))

    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

    # Make species list
    species = SpeciesList(
      p[1]
      for p in l_filter
      if p[0] >= location_filter_threshold
    )

    return species

  def analyze_file(
      self,
      file_path: Path,
      *,
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
    ) -> AnalysisResultV2p4:
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

    file_duration_seconds = librosa.get_duration(
      filename=str(file_path.absolute()), sr=self.sample_rate)

    use_species_filter = filter_species is not None
    if use_species_filter:
      species_filter_contains_unknown_species = not filter_species.issubset(self._species_list)
      if species_filter_contains_unknown_species:
        raise ValueError("filter_species")
      if len(filter_species) == 0:
        raise ValueError("filter_species")

    predictions = OrderedDict()

    # Process each chunk
    while offset < file_duration_seconds:
      # will resample to self.sample_rate
      audio_signal, _ = librosa.load(
        file_path,
        sr=self.sample_rate,
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

        audio_signal = bandpass_signal(audio_signal, self.sample_rate,
                                       bandpass_fmin, bandpass_fmax, self.sig_fmin, self.sig_fmax)

      samples = []
      timestamps = []

      chunks = split_signal(audio_signal, self.sample_rate, sig_length, sig_overlap, sig_minlen)

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
        prediction = self.__predict_audio(data)

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
          p_labels = zip(self._species_list, pred)

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

    result = AnalysisResultV2p4(
      file_path=file_path,
      model_version=MODEL_VERSION,
      file_duration_seconds=file_duration_seconds,
      use_bandpass=use_bandpass,
      bandpass_fmax=None,
      bandpass_fmin=None,
      applied_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=None,
      available_species=self._species_list,
      filtered_species=filter_species,
      language=self.language,
      batch_size=batch_size,
      file_splitting_duration=file_splitting_duration,
      min_confidence=min_confidence,
      sig_length=sig_length,
      sig_minlen=sig_minlen,
      sig_overlap=sig_overlap,
      start_time=start_time,
      end_time=end_time,
      duration_seconds=(end_time - start_time).total_seconds(),
      sample_rate=self.sample_rate,
      model_fmin=self.sig_fmin,
      model_fmax=self.sig_fmax,
      predictions=predictions,
    )

    if use_bandpass:
      result.bandpass_fmax = bandpass_fmax
      result.bandpass_fmin = bandpass_fmin

    if apply_sigmoid:
      result.sigmoid_sensitivity = sigmoid_sensitivity

    return result
