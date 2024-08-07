from operator import itemgetter
from pathlib import Path
from typing import Optional, OrderedDict, Set, Union

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.types import Language, Species, SpeciesPrediction, SpeciesPredictions
from birdnet.utils import (bandpass_signal, fillup_with_silence, flat_sigmoid,
                           get_birdnet_app_data_folder, itertools_batched,
                           load_audio_in_chunks_with_overlap)

AVAILABLE_LANGUAGES: Set[Language] = {
    "sv", "da", "hu", "th", "pt", "fr", "cs", "af", "en_uk", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "en_us", "no", "sk", "ar", "fi", "ro", "nl", "zh"
}


class ModelV2M4Base():
  """
  Model version 2.4

  This class represents version 2.4 of the model.
  """

  def __init__(self, /, *, language: Language = "en_us") -> None:
    """
    Initializes the ModelV2M4 instance.

    Parameters:
    -----------
    language : Language, optional, default="en_us"
        The language to use for the model's text processing. Must be one of the following available languages:
        "en_us", "en_uk", "sv", "da", "hu", "th", "pt", "fr", "cs", "af", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "no", "sk", "ar", "fi", "ro", "nl", "zh".

    Raises:
    -------
    ValueError
        If any of the input parameters are invalid.
    """
    if language not in AVAILABLE_LANGUAGES:
      raise ValueError(
        f"Language '{language}' is not available! Choose from: {', '.join(sorted(AVAILABLE_LANGUAGES))}.")

    self._language = language

    self._sig_fmin: int = 0
    self._sig_fmax: int = 15_000
    self._sample_rate = 48_000
    self._chunk_size_s: float = 3.0

    self._species_list: OrderedSet[Species] = OrderedSet()

    birdnet_app_data = get_birdnet_app_data_folder()
    self._model_version_folder = birdnet_app_data / "models" / "v2.4"

  @property
  def species(self) -> OrderedSet[Species]:
    return self._species_list

  def _predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError()

  def _predict_species_location(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError()

  def _predict_species_from_location(self, latitude: float, longitude: float, week: Optional[int]) -> npt.NDArray[np.float32]:
    assert -90 <= latitude <= 90
    assert -180 <= longitude <= 180
    assert week is None or (1 <= week <= 48)

    if week is None:
      week = -1
    assert week is not None

    sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

    prediction = self._predict_species_location(sample)

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
        "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0).")

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
