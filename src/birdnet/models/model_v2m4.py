import os
import zipfile
from operator import itemgetter
from pathlib import Path
from typing import Optional, OrderedDict, Set, Union

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet
from tensorflow.lite.python.interpreter import Interpreter

from birdnet.types import Language, Species, SpeciesPrediction, SpeciesPredictions
from birdnet.utils import (bandpass_signal, chunk_signal, download_file_tqdm, flat_sigmoid,
                           get_birdnet_app_data_folder, get_species_from_file, itertools_batched,
                           load_audio_file_in_parts)

AVAILABLE_LANGUAGES: Set[Language] = {
    "sv", "da", "hu", "th", "pt", "fr", "cs", "af", "en_uk", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "en_us", "no", "sk", "ar", "fi", "ro", "nl", "zh"
}


class Downloader():
  def __init__(self, app_storage: Path) -> None:
    self._version_path = app_storage / "models" / "v2.4"
    self._audio_model_path = self._version_path / "audio-model.tflite"
    self._meta_model_path = self._version_path / "meta-model.tflite"
    self._lang_path = self._version_path / "labels"

  @property
  def version_path(self) -> Path:
    return self._version_path

  @property
  def audio_model_path(self) -> Path:
    return self._audio_model_path

  @property
  def meta_model_path(self) -> Path:
    return self._meta_model_path

  def get_language_path(self, language: Language) -> Path:
    return self._lang_path / f"{language}.txt"

  def _check_model_files_exist(self) -> bool:
    model_is_downloaded = True
    model_is_downloaded &= self._audio_model_path.is_file()
    model_is_downloaded &= self._meta_model_path.is_file()
    model_is_downloaded &= self._lang_path.is_dir()
    for lang in AVAILABLE_LANGUAGES:
      model_is_downloaded &= self.get_language_path(lang).is_file()
    return model_is_downloaded

  def _download_model_files(self) -> None:
    dl_path = "https://tuc.cloud/index.php/s/45KmTcpHH8iDDA2/download/BirdNET_v2.4.zip"
    dl_size = 63092251
    self._version_path.mkdir(parents=True, exist_ok=True)

    zip_download_path = self._version_path / "download.zip"
    download_file_tqdm(dl_path, zip_download_path, download_size=dl_size,
                       description="Downloading model")

    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
      zip_ref.extractall(self._version_path)

    os.remove(zip_download_path)

  def ensure_model_is_available(self) -> None:
    if not self._check_model_files_exist():
      self._download_model_files()
      assert self._check_model_files_exist()


class ModelV2M4():
  """
  Model version 2.4

  This class represents version 2.4 of the model.
  """

  def __init__(self, tflite_num_threads: Optional[int] = 1, language: Language = "en_us") -> None:
    """
    Initializes the ModelV2M4 instance.

    Parameters:
    -----------
    tflite_num_threads : Optional[int], optional, default=1
        The number of threads to use for TensorFlow Lite operations. If None, the default threading behavior will be used.
        Must be a positive integer if specified.
    language : Language, optional, default="en_us"
        The language to use for the model's text processing. Must be one of the following available languages:
        "en_us", "en_uk", "sv", "da", "hu", "th", "pt", "fr", "cs", "af", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "no", "sk", "ar", "fi", "ro", "nl", "zh".

    Raises:
    -------
    ValueError
        If any of the input parameters are invalid.
    """
    super().__init__()

    if tflite_num_threads is not None and tflite_num_threads < 1:
      raise ValueError(
        "Value for 'tflite_num_threads' is invalid! It needs to be None or larger than zero.")

    if language not in AVAILABLE_LANGUAGES:
      raise ValueError(
        f"Language '{language}' is not available! Choose from: {', '.join(sorted(AVAILABLE_LANGUAGES))}.")

    self._language = language

    birdnet_app_data = get_birdnet_app_data_folder()
    downloader = Downloader(birdnet_app_data)
    downloader.ensure_model_is_available()

    self._sig_fmin: int = 0
    self._sig_fmax: int = 15_000
    self._sample_rate = 48_000
    self._chunk_size_s: float = 3.0
    self._chunk_overlap_s: float = 0.0
    self._min_chunk_size_s: float = 1.0

    self._species_list = get_species_from_file(
      downloader.get_language_path(language),
      encoding="utf8"
    )

    # Load TFLite model and allocate tensors.
    self._audio_interpreter = Interpreter(
      str(downloader.audio_model_path.absolute()), num_threads=tflite_num_threads)
    # Get input tensor index
    input_details = self._audio_interpreter.get_input_details()
    self._audio_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self._audio_interpreter.get_output_details()
    self._audio_output_layer_index = output_details[0]["index"]
    self._audio_interpreter.allocate_tensors()

    # Load TFLite model and allocate tensors.
    self._meta_interpreter = Interpreter(
      str(downloader.meta_model_path.absolute()), num_threads=tflite_num_threads)
    # Get input tensor index
    input_details = self._meta_interpreter.get_input_details()
    self._meta_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self._meta_interpreter.get_output_details()
    self._meta_output_layer_index = output_details[0]["index"]
    self._meta_interpreter.allocate_tensors()
    del downloader

  @property
  def species(self) -> OrderedSet[Species]:
    return self._species_list

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32
    # Same method as embeddings

    self._audio_interpreter.resize_tensor_input(self._audio_input_layer_index, batch.shape)
    self._audio_interpreter.allocate_tensors()

    # Make a prediction (Audio only for now)
    self._audio_interpreter.set_tensor(self._audio_input_layer_index, batch)
    self._audio_interpreter.invoke()
    prediction: npt.NDArray[np.float32] = self._audio_interpreter.get_tensor(
      self._audio_output_layer_index)

    return prediction

  def _predict_species_from_location(self, latitude: float, longitude: float, week: Optional[int]) -> npt.NDArray[np.float32]:
    assert -90 <= latitude <= 90
    assert -180 <= longitude <= 180
    assert week is None or (1 <= week <= 48)

    if week is None:
      week = -1
    assert week is not None

    sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

    # Run inference
    self._meta_interpreter.set_tensor(self._meta_input_layer_index, sample)
    self._meta_interpreter.invoke()

    prediction: npt.NDArray[np.float32] = self._meta_interpreter.get_tensor(
      self._meta_output_layer_index)[0]
    return prediction

  def predict_species_at_location_and_time(
      self,
      latitude: float,
      longitude: float,
      /,
      *,
      week: Optional[int] = None,
      min_confidence: float = 0.03
    ) -> SpeciesPrediction:
    """
    Predicts species at a specific geographic location and optionally during a specific week of the year.

    Parameters:
    -----------
    latitude : float
        The latitude of the location for species prediction. Must be in the interval [-90.0, 90.0].
    longitude : float
        The longitude of the location for species prediction. Must be in the interval [-180.0, 180.0].
    week : Optional[int], optional, default=None
        The week of the year for which to predict species. Must be in the interval [1, 48] if specified.
        If None, predictions are not limited to a specific week.
    min_confidence : float, optional, default=0.03
        Minimum confidence threshold for predictions to be considered valid. Must be in the interval [0, 1.0).

    Returns:
    --------
    SpeciesPrediction
        An ordered dictionary where:
        - The keys are species names (strings).
        - The values are confidence scores (floats) representing the likelihood of the species being present at the specified location and time.
        - The dictionary is sorted in descending order of confidence scores.

    Raises:
    -------
    ValueError
        If any of the input parameters are invalid.
    """

    if not -90 <= latitude <= 90:
      raise ValueError(
        "Value for 'latitude' is invalid! It needs to be in interval [-90.0, 90.0].")

    if not -180 <= longitude <= 180:
      raise ValueError(
        "Value for 'longitude' is invalid! It needs to be in interval [-180.0, 180.0].")

    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0, 1.0).")

    if week is not None and not (1 <= week <= 48):
      raise ValueError(
        "Value for 'week' is invalid! It needs to be either None or in interval [1, 48].")

    # Extract species from model
    l_filter = self._predict_species_from_location(latitude, longitude, week)

    prediction = (
      (species, score)
      for species, score in zip(self._species_list, l_filter)
      if score >= min_confidence
    )

    # Sort by filter value
    sorted_prediction = OrderedDict(
      sorted(prediction, key=itemgetter(1), reverse=True)
    )

    return sorted_prediction

  def predict_species_within_audio_file(
      self,
      audio_file: Path,
      /,
      *,
      min_confidence: float = 0.1,
      batch_size: int = 1,
      use_bandpass: bool = True,
      bandpass_fmin: Optional[int] = 0,
      bandpass_fmax: Optional[int] = 15_000,
      apply_sigmoid: bool = True,
      sigmoid_sensitivity: Optional[float] = 1.0,
      filter_species: Optional[Union[Set[Species], OrderedSet[Species]]] = None,
      file_splitting_duration_s: float = 600,
    ) -> SpeciesPredictions:
    """
    Predicts species within an audio file.

    Parameters:
    -----------
    audio_file : Path
        The path to the audio file for species prediction.
    min_confidence : float, optional, default=0.1
        Minimum confidence threshold for predictions to be considered valid.
    batch_size : int, optional, default=1
        Number of audio samples to process in a batch.
    use_bandpass : bool, optional, default=True
        Whether to apply a bandpass filter to the audio.
    bandpass_fmin : Optional[int], optional, default=0
        Minimum frequency for the bandpass filter (in Hz). Ignored if `use_bandpass` is False.
    bandpass_fmax : Optional[int], optional, default=15_000
        Maximum frequency for the bandpass filter (in Hz). Ignored if `use_bandpass` is False.
    apply_sigmoid : bool, optional, default=True
        Whether to apply a sigmoid function to the model outputs.
    sigmoid_sensitivity : Optional[float], optional, default=1.0
        Sensitivity parameter for the sigmoid function. Must be in the interval [0.5, 1.5]. Ignored if `apply_sigmoid` is False.
    filter_species : Optional[Set[Species]], optional
        A set of species to filter the predictions. If None, no filtering is applied.
    file_splitting_duration_s : float, optional, default=600
        Duration in seconds for splitting the audio file into smaller segments for processing.

    Returns:
    --------
    SpeciesPredictions
        The predictions of species within the audio file. This is an ordered dictionary where:
        - The keys are time intervals (tuples of start and end times in seconds) representing segments of the audio file.
        - The values are ordered dictionaries where:
            - The keys are species names (strings).
            - The values are confidence scores (floats) representing the likelihood of the presence of the species in the given time interval.

    Raises:
    -------
    ValueError
        If any of the input parameters are invalid.
    """

    if not audio_file.is_file():
      raise ValueError(
        "Value for 'audio_file' is invalid! It needs to be a path to an existing audio file.")

    if batch_size < 1:
      raise ValueError(
        "Value for 'batch_size' is invalid! It needs to be larger than zero.")

    if file_splitting_duration_s < self._chunk_size_s:
      raise ValueError(
        f"Value for 'file_splitting_duration_s' is invalid! It needs to be larger than or equal to {self._chunk_size_s}.")

    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0, 1.0).")

    if apply_sigmoid:
      if sigmoid_sensitivity is None:
        raise ValueError("Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!")
      if not 0.5 <= sigmoid_sensitivity <= 1.5:
        raise ValueError(
          "Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval [0.5, 1.5].")

    use_species_filter = filter_species is not None and len(filter_species) > 0
    if use_species_filter:
      assert filter_species is not None  # added for mypy
      species_filter_contains_unknown_species = not filter_species.issubset(self._species_list)
      if species_filter_contains_unknown_species:
        raise ValueError(
          f"At least one species defined in 'filter_species' is invalid! They need to be known species, e.g., {', '.join(self._species_list[:3])}")

    predictions = OrderedDict()

    for audio_signal_part in load_audio_file_in_parts(
      audio_file, self._sample_rate, file_splitting_duration_s
    ):
      if use_bandpass:
        if bandpass_fmin is None:
          raise ValueError("Value for 'bandpass_fmin' is required if 'use_bandpass==True'!")
        if bandpass_fmax is None:
          raise ValueError("Value for 'bandpass_fmax' is required if 'use_bandpass==True'!")

        if bandpass_fmin < 0:
          raise ValueError("Value for 'bandpass_fmin' is invalid! It needs to be larger than zero.")

        if bandpass_fmax <= bandpass_fmin:
          raise ValueError(
            "Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'.")

        audio_signal_part = bandpass_signal(audio_signal_part, self._sample_rate,
                                            bandpass_fmin, bandpass_fmax, self._sig_fmin, self._sig_fmax)

      chunked_signal = chunk_signal(
        audio_signal_part, self._sample_rate, self._chunk_size_s, self._chunk_overlap_s, self._min_chunk_size_s
      )

      for batch_of_chunks in itertools_batched(chunked_signal, batch_size):
        batch = np.array(list(map(itemgetter(2), batch_of_chunks)), np.float32)
        predicted_species = self._predict_species(batch)

        if apply_sigmoid:
          assert sigmoid_sensitivity is not None
          predicted_species = flat_sigmoid(
            predicted_species,
            sensitivity=-sigmoid_sensitivity,
          )

        for i, (chunk_start, chunk_end, _) in enumerate(batch_of_chunks):
          prediction = predicted_species[i]

          labeled_prediction = (
            (species, score)
            for species, score in zip(self._species_list, prediction)
            if score >= min_confidence
          )

          if use_species_filter:
            assert filter_species is not None  # added for mypy
            labeled_prediction = (
              (species, score)
              for species, score in labeled_prediction
              if species in filter_species
            )

          # Sort by score
          sorted_prediction = OrderedDict(
            sorted(labeled_prediction, key=itemgetter(1), reverse=True)
          )
          assert (chunk_start, chunk_end) not in predictions
          predictions[(chunk_start, chunk_end)] = sorted_prediction

    return predictions
