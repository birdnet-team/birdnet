import os
import zipfile
from logging import getLogger
from pathlib import Path
from typing import List, Optional, cast

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import Tensor

from birdnet.models.model_v2m4_base import AVAILABLE_LANGUAGES, ModelV2M4Base
from birdnet.types import Language
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


class ModelV2M4Protobuf(ModelV2M4Base):
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
        This parameter allows specifying a custom device on which computations should be performed. If custom_device is not specified (i.e., it has the default value None), the program will attempt to use a GPU (e.g., /device:GPU:0) by default. If no GPU is available, it will fall back to using the CPU. By specifying a device string such as /device:GPU:0 or /device:CPU:0, the user can explicitly choose the device on which operations should be executed.

    Raises:
    -------
    ValueError
        If any of the input parameters are invalid.
    """
    super().__init__(language=language)

    device: tf.config.LogicalDevice
    if custom_device is None:
      device = try_get_gpu_otherwise_return_cpu()
    else:
      matched_device = None
      available_devices: List[tf.config.LogicalDevice] = tf.config.list_logical_devices()

      for logical_device in available_devices:
        if logical_device.name == custom_device:
          matched_device = logical_device
          break
      if matched_device is None:
        raise ValueError(
          f"Device '{custom_device}' doesn't exist. Please select one of the following existing device names: {', '.join(d.name for d in available_devices)}.")
      device = matched_device
    self._device = device

    logger = getLogger(__name__)
    logger.info(f"Using device: {self._device.name}")

    model_folder = self._model_version_folder / "Protobuf"
    downloader = DownloaderProtobuf(model_folder)
    downloader.ensure_model_is_available()

    self._species_list = get_species_from_file(
      downloader.get_language_path(language),
      encoding="utf8"
    )

    self._audio_model = tf.saved_model.load(downloader.audio_model_path.absolute())
    self._meta_model = tf.saved_model.load(downloader.meta_model_path.absolute())
    del downloader

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32
    with tf.device(self._device):
      prediction: Tensor = self._audio_model.basic(batch)["scores"]
    prediction_np: npt.NDArray[np.float32] = prediction.numpy()
    return prediction_np

  def _predict_species_location(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert sample.dtype == np.float32
    with tf.device(self._device):
      prediction: Tensor = self._meta_model(sample)
    prediction = tf.squeeze(prediction)
    prediction_np: npt.NDArray[np.float32] = prediction.numpy()
    return prediction_np
