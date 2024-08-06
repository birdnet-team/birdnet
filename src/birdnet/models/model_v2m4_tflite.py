import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from tensorflow.lite.python.interpreter import Interpreter

from birdnet.models.model_v2m4_base import AVAILABLE_LANGUAGES, ModelV2M4Base
from birdnet.types import Language
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


class ModelV2M4TFLite(ModelV2M4Base):
  """
  Model version 2.4

  This class represents version 2.4 of the model.
  """

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
    super().__init__(language=language)

    if tflite_num_threads is not None and tflite_num_threads < 1:
      raise ValueError(
        "Value for 'tflite_num_threads' is invalid! It needs to be None or larger than zero.")

    model_folder = self._model_version_folder / "TFLite"
    downloader = DownloaderTFLite(model_folder)
    downloader.ensure_model_is_available()

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

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32

    self._audio_interpreter.resize_tensor_input(self._audio_input_layer_index, batch.shape)
    self._audio_interpreter.allocate_tensors()

    self._audio_interpreter.set_tensor(self._audio_input_layer_index, batch)
    self._audio_interpreter.invoke()
    prediction: npt.NDArray[np.float32] = self._audio_interpreter.get_tensor(
      self._audio_output_layer_index)

    return prediction

  def _predict_species_location(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert sample.dtype == np.float32
    self._meta_interpreter.set_tensor(self._meta_input_layer_index, sample)
    self._meta_interpreter.invoke()
    prediction: npt.NDArray[np.float32] = self._meta_interpreter.get_tensor(
      self._meta_output_layer_index)[0]
    return prediction
