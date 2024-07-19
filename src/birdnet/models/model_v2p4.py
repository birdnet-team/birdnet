import os
import zipfile
from operator import itemgetter
from pathlib import Path
from typing import Optional, OrderedDict, Set

import numpy as np
import numpy.typing as npt
import tflite_runtime.interpreter as tflite

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


class ModelV2p4():
  def __init__(self, tflite_threads: int = 1, language: Language = "en_us") -> None:
    super().__init__()
    if language not in AVAILABLE_LANGUAGES:
      raise ValueError(
        f"Language '{language}' is not available! Choose from: {','.join(sorted(AVAILABLE_LANGUAGES))}")

    self.language = language

    birdnet_app_data = get_birdnet_app_data_folder()
    downloader = Downloader(birdnet_app_data)
    downloader.ensure_model_is_available()

    self.sig_fmin: int = 0
    self.sig_fmax: int = 15_000
    self.sample_rate = 48_000
    self.chunk_size_s: float = 3.0
    self.chunk_overlap_s: float = 0.0
    self.min_chunk_size_s: float = 1.0

    self._species_list = get_species_from_file(
      downloader.get_language_path(language), 
      encoding="utf8"
    )
    # [line.split(",")[1] for line in labels]

    # Load TFLite model and allocate tensors.
    self.audio_interpreter = tflite.Interpreter(
      str(downloader.audio_model_path.absolute()), num_threads=tflite_threads)
    # Get input tensor index
    input_details = self.audio_interpreter.get_input_details()
    self.audio_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self.audio_interpreter.get_output_details()
    self.audio_output_layer_index = output_details[0]["index"]
    self.audio_interpreter.allocate_tensors()

    # Load TFLite model and allocate tensors.
    self.meta_interpreter = tflite.Interpreter(
      str(downloader.meta_model_path.absolute()), num_threads=tflite_threads)
    # Get input tensor index
    input_details = self.meta_interpreter.get_input_details()
    self.meta_input_layer_index = input_details[0]["index"]
    # Get classification output
    output_details = self.meta_interpreter.get_output_details()
    self.meta_output_layer_index = output_details[0]["index"]
    self.meta_interpreter.allocate_tensors()
    del downloader

  @property
  def species(self):
    return self._species_list

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32
    # Same method as embeddings

    self.audio_interpreter.resize_tensor_input(self.audio_input_layer_index, batch.shape)
    self.audio_interpreter.allocate_tensors()

    # Make a prediction (Audio only for now)
    self.audio_interpreter.set_tensor(self.audio_input_layer_index, batch)
    self.audio_interpreter.invoke()
    prediction: npt.NDArray[np.float32] = self.audio_interpreter.get_tensor(
      self.audio_output_layer_index)

    return prediction

  def _predict_species_from_location(self, latitude: float, longitude: float, week: int) -> npt.NDArray[np.float32]:
    assert -90 <= latitude <= 90
    assert -180 <= longitude <= 180
    assert 1 <= week <= 48 or week == -1

    sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

    # Run inference
    self.meta_interpreter.set_tensor(self.meta_input_layer_index, sample)
    self.meta_interpreter.invoke()

    prediction: npt.NDArray[np.float32] = self.meta_interpreter.get_tensor(self.meta_output_layer_index)[
        0]
    return prediction

  def predict_species_at_location_and_time(self, latitude: float, longitude: float, *, week: int = -1, min_confidence: float = 0.03) -> SpeciesPrediction:
    """Predict a species set.

    Uses the model to predict the species list for the given coordinates and filters by threshold.

    Args:
        week: The week of the year [1-48]. Use -1 for year-round.
        threshold: Only values above or equal to threshold will be shown.

    Returns:
        A set of all eligible species including their scores.
    """

    if not -90 <= latitude <= 90:
      raise ValueError("latitude")

    if not -180 <= longitude <= 180:
      raise ValueError("longitude")

    if not 0 <= min_confidence < 1.0:
      raise ValueError("min_confidence")

    if not (1 <= week <= 48 or week == -1):
      raise ValueError("week")

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

  def predict_species_in_audio_file(
      self,
      audio_file: Path,
      *,
      min_confidence: float = 0.1,
      batch_size: int = 1,
      use_bandpass: bool = True,
      bandpass_fmin: Optional[int] = 0,
      bandpass_fmax: Optional[int] = 15_000,
      apply_sigmoid: bool = True,
      sigmoid_sensitivity: Optional[float] = 1.0,
      filter_species: Optional[Set[Species]] = None,
      file_splitting_duration_s: float = 600,
    ) -> SpeciesPredictions:
    """
    sig_minlen: Define minimum length of audio chunk for prediction; chunks shorter than 3 seconds will be padded with zeros
    """

    if batch_size < 1:
      raise ValueError("batch_size")

    if file_splitting_duration_s <= 0:
      raise ValueError("file_splitting_duration_s")

    if not 0 <= min_confidence < 1.0:
      raise ValueError("min_confidence")

    if apply_sigmoid:
      if sigmoid_sensitivity is None:
        raise ValueError("sigmoid_sensitivity")
      if not 0.5 <= sigmoid_sensitivity <= 1.5:
        raise ValueError("sigmoid_sensitivity")

    use_species_filter = filter_species is not None
    if use_species_filter:
      assert filter_species is not None  # added for mypy
      species_filter_contains_unknown_species = not filter_species.issubset(self._species_list)
      if species_filter_contains_unknown_species:
        raise ValueError("filter_species")
      if len(filter_species) == 0:
        raise ValueError("filter_species")

    predictions = OrderedDict()

    for audio_signal_part in load_audio_file_in_parts(
      audio_file, self.sample_rate, file_splitting_duration_s
    ):
      if use_bandpass:
        if bandpass_fmin is None:
          raise ValueError("bandpass_fmin")
        if bandpass_fmax is None:
          raise ValueError("bandpass_fmax")

        audio_signal_part = bandpass_signal(audio_signal_part, self.sample_rate,
                                            bandpass_fmin, bandpass_fmax, self.sig_fmin, self.sig_fmax)

      chunked_signal = chunk_signal(
        audio_signal_part, self.sample_rate, self.chunk_size_s, self.chunk_overlap_s, self.min_chunk_size_s
      )

      for batch_of_chunks in itertools_batched(chunked_signal, batch_size):
        batch = np.array(list(map(itemgetter(2), batch_of_chunks)), np.float32)
        predicted_species = self._predict_species(batch)

        # Logits or sigmoid activations?
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
