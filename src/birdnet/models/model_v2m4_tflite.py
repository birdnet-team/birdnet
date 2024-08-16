import os
import zipfile
from pathlib import Path
from typing import Optional, Set, Union

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet
from tensorflow.lite.python.interpreter import Interpreter

from birdnet.models.model_v2m4_base import (AVAILABLE_LANGUAGES, AudioModelBaseV2M4,
                                            MetaModelBaseV2M4, get_internal_version_app_data_folder,
                                            validate_language)
from birdnet.types import Language, Species, SpeciesPrediction, SpeciesPredictions
from birdnet.utils import download_file_tqdm, get_species_from_file


class DownloaderTFLite():
  def __init__(self, parent_folder: Path) -> None:
    self._version_path = parent_folder
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
    if self._audio_model_path.is_file():
      file_stats = os.stat(self._audio_model_path)
      audio_is_newest_version = file_stats.st_size == 51726412
      model_is_downloaded &= audio_is_newest_version

    model_is_downloaded &= self._meta_model_path.is_file()
    if self._meta_model_path.is_file():
      file_stats = os.stat(self._meta_model_path)
      meta_is_newest_version = file_stats.st_size == 29526096
      model_is_downloaded &= meta_is_newest_version

    model_is_downloaded &= self._lang_path.is_dir()
    for lang in AVAILABLE_LANGUAGES:
      model_is_downloaded &= self.get_language_path(lang).is_file()
    return model_is_downloaded

  def _download_model_files(self) -> None:
    dl_path = "https://tuc.cloud/index.php/s/45KmTcpHH8iDDA2/download/BirdNET_v2.4.zip"
    dl_size = 76823623
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


class MetaModelV2M4TFLite(MetaModelBaseV2M4):
  def __init__(self, model_path: Path, species_list: OrderedSet[str], tflite_num_threads: Optional[int]) -> None:
    super().__init__(species_list)
    assert tflite_num_threads is None or (tflite_num_threads >= 1)

    # Load TFLite model and allocate tensors.
    self._meta_interpreter = Interpreter(
      str(model_path.absolute()), num_threads=tflite_num_threads)
    # Get input tensor index
    input_details = self._meta_interpreter.get_input_details()
    self._meta_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self._meta_interpreter.get_output_details()
    self._meta_output_layer_index = output_details[0]["index"]
    self._meta_interpreter.allocate_tensors()

  def _predict_species_location(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert sample.dtype == np.float32
    self._meta_interpreter.set_tensor(self._meta_input_layer_index, sample)
    self._meta_interpreter.invoke()
    prediction: npt.NDArray[np.float32] = self._meta_interpreter.get_tensor(
      self._meta_output_layer_index)[0]
    return prediction


class AudioModelV2M4TFLite(AudioModelBaseV2M4):
  def __init__(self, model_path: Path, species_list: OrderedSet[str], tflite_num_threads: Optional[int]) -> None:
    super().__init__(species_list)
    assert tflite_num_threads is None or (tflite_num_threads >= 1)

    # Load TFLite model and allocate tensors.
    self._audio_interpreter = Interpreter(
      str(model_path.absolute()), num_threads=tflite_num_threads)
    # Get input tensor index
    input_details = self._audio_interpreter.get_input_details()
    self._audio_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self._audio_interpreter.get_output_details()
    self._audio_output_layer_index = output_details[0]["index"]
    self._audio_interpreter.allocate_tensors()

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32

    self._audio_interpreter.resize_tensor_input(self._audio_input_layer_index, batch.shape)
    self._audio_interpreter.allocate_tensors()

    self._audio_interpreter.set_tensor(self._audio_input_layer_index, batch)
    self._audio_interpreter.invoke()
    prediction: npt.NDArray[np.float32] = self._audio_interpreter.get_tensor(
      self._audio_output_layer_index)

    return prediction


class ModelV2M4TFLite():
  def __init__(self, /, *, tflite_num_threads: Optional[int] = 1, language: Language = "en_us") -> None:
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
    if tflite_num_threads is not None and tflite_num_threads < 1:
      raise ValueError(
        "Value for 'tflite_num_threads' is invalid! It needs to be None or larger than zero.")

    validate_language(language)

    model_folder = get_internal_version_app_data_folder() / "TFLite"
    downloader = DownloaderTFLite(model_folder)
    downloader.ensure_model_is_available()

    species_list = get_species_from_file(
      downloader.get_language_path(language),
      encoding="utf8"
    )

    self._meta_model = MetaModelV2M4TFLite(
      downloader.meta_model_path, species_list, tflite_num_threads)
    self._audio_model = AudioModelV2M4TFLite(
      downloader.audio_model_path, species_list, tflite_num_threads)
    assert self._meta_model.species == self._audio_model.species
    # self.predict_species_within_audio_file.__doc__ = self._audio_model.predict_species_within_audio_file.__doc__

  @property
  def species(self) -> OrderedSet[Species]:
    return self._audio_model.species

  def predict_species_within_audio_file(
      self,
      audio_file: Path,
      /,
      *,
      min_confidence: float = 0.1,
      batch_size: int = 1,
      chunk_overlap_s: float = 0.0,
      use_bandpass: bool = True,
      bandpass_fmin: Optional[int] = 0,
      bandpass_fmax: Optional[int] = 15_000,
      apply_sigmoid: bool = True,
      sigmoid_sensitivity: Optional[float] = 1.0,
      filter_species: Optional[Union[Set[Species], OrderedSet[Species]]] = None,
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
    chunk_overlap_s : float, optional, default=0.0
        Overlapping of chunks in seconds. Must be in the interval [0.0, 3.0).
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
    return self._audio_model.predict_species_within_audio_file(
      audio_file,
      min_confidence=min_confidence,
      batch_size=batch_size,
      chunk_overlap_s=chunk_overlap_s,
      use_bandpass=use_bandpass,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      filter_species=filter_species,
    )

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
    return self._meta_model.predict_species_at_location_and_time(
      latitude,
      longitude,
      week=week,
      min_confidence=min_confidence,
    )
