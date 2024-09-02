from operator import itemgetter
from pathlib import Path
from typing import Generator, Optional, Set, Tuple, Union

import numpy as np
from ordered_set import OrderedSet

from birdnet.models.v2m4.model_v2m4_base import AudioModelBaseV2M4
from birdnet.models.v2m4.model_v2m4_protobuf import AudioModelV2M4Protobuf
from birdnet.types import Species, SpeciesPrediction, TimeInterval
from birdnet.utils import (bandpass_signal, fillup_with_silence, flat_sigmoid, itertools_batched,
                           load_audio_in_chunks_with_overlap)


def predict_species_within_audio_file(
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
    custom_model: Optional[AudioModelBaseV2M4] = None,
  ) -> Generator[Tuple[TimeInterval, SpeciesPrediction], None, None]:
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

  Yields:
  -------
  Tuple[TimeInterval, SpeciesPrediction]
      Each item yielded by the generator is a tuple, where:
      - The first element is a time interval (a tuple of start and end times in seconds) representing a segment of the audio file.
      - The second element is an ordered dictionary where:
          - The keys are species names (strings).
          - The values are confidence scores (floats) indicating the likelihood of the species being present within the corresponding time interval.

  Raises:
  -------
  ValueError
      If any of the input parameters are invalid.
  """

  model: AudioModelBaseV2M4
  if custom_model is None:
    model = AudioModelV2M4Protobuf()
  else:
    model = custom_model

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
    species_filter_contains_unknown_species = not filter_species.issubset(model.species)
    if species_filter_contains_unknown_species:
      raise ValueError(
        f"At least one species defined in 'filter_species' is invalid! They need to be known species, e.g., {', '.join(model.species[:3])}")

  chunked_audio = load_audio_in_chunks_with_overlap(
    audio_file,
    chunk_duration_s=model.chunk_size_s,
    overlap_duration_s=chunk_overlap_s,
    target_sample_rate=model.sample_rate,
  )

  # fill last chunk with silence up to chunksize if it is smaller than 3s
  chunk_sample_size = round(model.sample_rate * model.chunk_size_s)
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
      (start, end, bandpass_signal(
        chunk,
        model.sample_rate,
        bandpass_fmin,
        bandpass_fmax,
        model.sig_fmin,
        model.sig_fmax,
      ))
      for start, end, chunk in chunked_audio
    )
    chunked_audio = chunked_audio_bandpassed

  batches = itertools_batched(chunked_audio, batch_size)

  for batch_of_chunks in batches:
    batch = np.array(list(map(itemgetter(2), batch_of_chunks)), np.float32)
    predicted_species = model.predict_species(batch)

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
        for species, score in zip(model.species, prediction)
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
      sorted_prediction = SpeciesPrediction(
        sorted(
          labeled_prediction,
          key=lambda species_score: (species_score[1] * -1, species_score[0]),
          reverse=False,
        )
      )
      time_interval: TimeInterval = (chunk_start, chunk_end)
      yield time_interval, sorted_prediction
