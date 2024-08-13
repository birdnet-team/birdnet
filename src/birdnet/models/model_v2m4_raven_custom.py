from logging import getLogger
from operator import itemgetter
from pathlib import Path
from typing import List, Optional, OrderedDict, Set, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from ordered_set import OrderedSet
from tensorflow import Tensor

from birdnet.models.model_v2m4_protobuf import (check_protobuf_model_files_exist,
                                                try_get_gpu_otherwise_return_cpu)
from birdnet.types import Species, SpeciesPredictions
from birdnet.utils import (bandpass_signal, fillup_with_silence, flat_sigmoid,
                           get_species_from_file, itertools_batched,
                           load_audio_in_chunks_with_overlap)


class CustomRavenParser():
  def __init__(self, classifier_folder: Path, classifier_name: str) -> None:
    self._audio_model_path = classifier_folder / f"{classifier_name}"
    self._label_path = classifier_folder / f"{classifier_name}" / "labels" / "label_names.csv"

  @property
  def audio_model_path(self) -> Path:
    return self._audio_model_path

  @property
  def language_path(self) -> Path:
    return self._label_path

  def check_model_files_exist(self) -> bool:
    model_is_available = True
    model_is_available &= self._audio_model_path.is_dir()
    model_is_available &= self._label_path.is_file()
    model_is_available &= check_protobuf_model_files_exist(self.audio_model_path)
    return model_is_available


class CustomModelV2M4Raven():
  """
  Model version 2.4

  This class represents version 2.4 of the model.
  """

  def __init__(self, classifier_folder: Path, classifier_name: str, /, *, custom_device: Optional[str] = None) -> None:

    parser = CustomRavenParser(classifier_folder, classifier_name)
    if not parser.check_model_files_exist():
      raise ValueError(
        f"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid raven classifier which has the name '{classifier_name}'!")

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

    self._sig_fmin: int = 0
    self._sig_fmax: int = 15_000
    self._sample_rate = 48_000
    self._chunk_size_s: float = 3.0

    self._species_list = get_species_from_file(
      parser.language_path,
      encoding="utf8"
    )

    self._audio_model = tf.saved_model.load(parser.audio_model_path.absolute())
    del parser

  @property
  def species(self) -> OrderedSet[Species]:
    return self._species_list

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert batch.dtype == np.float32
    with tf.device(self._device):
      prediction: Tensor = self._audio_model.basic(batch)["scores"]
    prediction_np: npt.NDArray[np.float32] = prediction.numpy()
    return prediction_np

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

    if not audio_file.is_file():
      raise ValueError(
        "Value for 'audio_file' is invalid! It needs to be a path to an existing audio file.")

    if batch_size < 1:
      raise ValueError(
        "Value for 'batch_size' is invalid! It needs to be larger than zero.")

    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0).")

    if not 0 <= chunk_overlap_s < 3:
      raise ValueError(
        "Value for 'chunk_overlap_s' is invalid! It needs to be in interval [0.0, 3.0).")

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

    chunked_audio = load_audio_in_chunks_with_overlap(audio_file,
                                                      chunk_duration_s=self._chunk_size_s, overlap_duration_s=chunk_overlap_s, target_sample_rate=self._sample_rate)

    # fill last chunk with silence up to chunksize if it is smaller than 3s
    chunk_sample_size = round(self._sample_rate * self._chunk_size_s)
    chunked_audio = (
      (start, end, fillup_with_silence(chunk, chunk_sample_size))
      for start, end, chunk in chunked_audio
    )

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

      chunked_audio_bandpassed = (
        (start, end, bandpass_signal(chunk, self._sample_rate, bandpass_fmin,
                                     bandpass_fmax, self._sig_fmin, self._sig_fmax))
        for start, end, chunk in chunked_audio
      )
      chunked_audio = chunked_audio_bandpassed

    batches = itertools_batched(chunked_audio, batch_size)

    for batch_of_chunks in batches:
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

        # Sort by score then by name
        sorted_prediction = OrderedDict(
          sorted(labeled_prediction, key=lambda species_score: (
            species_score[1] * -1, species_score[0]), reverse=False)
        )
        key = (chunk_start, chunk_end)
        assert key not in predictions
        predictions[key] = sorted_prediction

    return predictions
