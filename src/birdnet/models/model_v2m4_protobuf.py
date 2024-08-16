import os
import zipfile
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Set, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from ordered_set import OrderedSet
from tensorflow import Tensor

from birdnet.models.model_v2m4_base import (AVAILABLE_LANGUAGES, AudioModelBaseV2M4,
                                            MetaModelBaseV2M4, get_internal_version_app_data_folder,
                                            validate_language)
from birdnet.types import Language, Species, SpeciesPrediction, SpeciesPredictions
from birdnet.utils import download_file_tqdm, get_species_from_file


def check_protobuf_model_files_exist(folder: Path) -> bool:
  exists = True
  exists &= (folder / "saved_model.pb").is_file()
  exists &= (folder / "variables").is_dir()
  exists &= (folder / "variables" / "variables.data-00000-of-00001").is_file()
  exists &= (folder / "variables" / "variables.index").is_file()
  return exists


class DownloaderProtobuf():
  def __init__(self, parent_folder: Path) -> None:
    self._version_path = parent_folder
    self._audio_model_path = self._version_path / "audio-model"
    self._meta_model_path = self._version_path / "meta-model"
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
    model_is_downloaded &= self._audio_model_path.is_dir()
    model_is_downloaded &= self._meta_model_path.is_dir()
    model_is_downloaded &= self._lang_path.is_dir()
    for lang in AVAILABLE_LANGUAGES:
      model_is_downloaded &= self.get_language_path(lang).is_file()
    model_is_downloaded &= check_protobuf_model_files_exist(self.audio_model_path)
    model_is_downloaded &= check_protobuf_model_files_exist(self._meta_model_path)
    return model_is_downloaded

  def _download_model_files(self) -> None:
    dl_path = "https://tuc.cloud/index.php/s/ko6DA29EMwLBe3c/download/BirdNET_v2.4_protobuf.zip"
    dl_size = 124524452
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


def try_get_gpu_otherwise_return_cpu() -> tf.config.LogicalDevice:
  all_gpus = tf.config.list_logical_devices('GPU')
  if len(all_gpus) > 0:
    first_gpu = all_gpus[0]
    return first_gpu
  all_cpus = tf.config.list_logical_devices('CPU')
  if len(all_cpus) == 0:
    raise Exception("No CPU found!")
  first_cpu = all_cpus[0]
  return first_cpu


def get_custom_device(device_name: str) -> tf.config.LogicalDevice:
  matched_device: tf.config.LogicalDevice = None
  available_devices: List[tf.config.LogicalDevice] = tf.config.list_logical_devices()

  for logical_device in available_devices:
    if logical_device.name == device_name:
      matched_device = logical_device
      break
  if matched_device is None:
    raise ValueError(
      f"Device '{device_name}' doesn't exist. Please select one of the following existing device names: {', '.join(d.name for d in available_devices)}.")
  return matched_device


class MetaModelV2M4Protobuf(MetaModelBaseV2M4):
  def __init__(self, model_path: Path, species_list: OrderedSet[str], device: tf.config.LogicalDevice) -> None:
    super().__init__(species_list)
    self._device = device
    logger = getLogger(__name__)
    logger.info(f"Using device: {self._device.name}")
    self._meta_model = tf.saved_model.load(model_path.absolute())

  def _predict_species_location(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert sample.dtype == np.float32
    with tf.device(self._device):
      prediction: Tensor = self._meta_model(sample)
    prediction = tf.squeeze(prediction)
    prediction_np: npt.NDArray[np.float32] = prediction.numpy()
    return prediction_np


class AudioModelV2M4Protobuf(AudioModelBaseV2M4):
  def __init__(self, model_path: Path, species_list: OrderedSet[str], device: tf.config.LogicalDevice) -> None:
    super().__init__(species_list)
    self._device = device
    logger = getLogger(__name__)
    logger.info(f"Using device: {self._device.name}")
    self._audio_model = tf.saved_model.load(model_path.absolute())

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32
    with tf.device(self._device):
      prediction: Tensor = self._audio_model.basic(batch)["scores"]
    prediction_np: npt.NDArray[np.float32] = prediction.numpy()
    return prediction_np


class ModelV2M4Protobuf():
  """
  Model version 2.4

  This class represents version 2.4 of the model.
  """

  def __init__(self, /, *, language: Language = "en_us", custom_device: Optional[str] = None) -> None:
    """
    Initializes the ModelV2M4 instance.

    Parameters:
    -----------
    language : Language, optional, default="en_us"
        The language to use for the model's text processing. Must be one of the following available languages:
        "en_us", "en_uk", "sv", "da", "hu", "th", "pt", "fr", "cs", "af", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "no", "sk", "ar", "fi", "ro", "nl", "zh".
    custom_device : str, optional, default=None
        This parameter allows specifying a custom device on which computations should be performed. If custom_device is not specified (i.e., it has the default value None), the program will attempt to use a GPU (e.g., "/device:GPU:0") by default. If no GPU is available, it will fall back to using the CPU. By specifying a device string such as "/device:GPU:0" or "/device:CPU:0", the user can explicitly choose the device on which operations should be executed.

    Raises:
    -------
    ValueError
        If any of the input parameters are invalid.
    """

    validate_language(language)

    model_folder = get_internal_version_app_data_folder() / "Protobuf"
    downloader = DownloaderProtobuf(model_folder)
    downloader.ensure_model_is_available()

    species_list = get_species_from_file(
      downloader.get_language_path(language),
      encoding="utf8"
    )

    device: tf.config.LogicalDevice
    if custom_device is None:
      device = try_get_gpu_otherwise_return_cpu()
    else:
      device = get_custom_device(custom_device)

    self._meta_model = MetaModelV2M4Protobuf(downloader.meta_model_path, species_list, device)
    self._audio_model = AudioModelV2M4Protobuf(downloader.audio_model_path, species_list, device)

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
