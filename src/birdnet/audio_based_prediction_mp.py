from collections import OrderedDict
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Callable, Generator, List, Optional, Set, Tuple, Union

from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.audio_based_prediction import predict_species_within_audio_file_core
from birdnet.models.v2m4.model_v2m4_base import AudioModelBaseV2M4
from birdnet.models.v2m4.model_v2m4_tflite import AudioModelV2M4TFLite, AudioModelV2M4TFLiteBase
from birdnet.types import Species, SpeciesPredictions


def predict_species_within_audio_files_mp(
    audio_files: List[Path],
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
    species_filter: Optional[Union[Set[Species], OrderedSet[Species]]] = None,
    custom_n_jobs: Optional[int] = None,
    custom_model: Optional[AudioModelV2M4TFLiteBase] = None,
    maxtasksperchild: Optional[int] = None,
    silent: bool = False,
  ) -> Generator[Tuple[Path, SpeciesPredictions], None, None]:
  """
  Predicts species within audio files using multiprocessing.

  Parameters:
  -----------
  audio_files : List[Path]
      A list of paths to the audio files for species prediction.
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
  species_filter : Optional[Union[Set[Species], OrderedSet[Species]]], optional
      A set or ordered set of species to filter the predictions. If None, no filtering is applied.
  custom_n_jobs : Optional[int], optional
      The number of jobs to run in parallel for multiprocessing. If None, it will use the number of CPUs available.
  custom_model : Optional[AudioModelV2M4TFLiteBase], optional
      Custom model to be used for species prediction. If None, the default TFLite model is used.
  maxtasksperchild : Optional[int], optional
      Maximum number of tasks per child process.
  silent : bool, optional, default=False
      Whether to disable the progress bar.

  Yields:
  -------
  Tuple[Path, SpeciesPredictions]
      Each item yielded by the generator is a tuple, where:
      - The first element is the path to the audio file being processed.
      - The second element is a `SpeciesPredictions` object containing:
          - The predictions made for each segment of the audio file.
          - Each prediction includes a time interval and a dictionary with species names as keys and their confidence scores as values.

  Raises:
  -------
  ValueError
      If any of the input parameters are invalid.
  """

  if not 0 <= min_confidence < 1.0:
    raise ValueError(
      "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0).")

  if not 0 <= chunk_overlap_s < 3:
    raise ValueError(
      "Value for 'chunk_overlap_s' is invalid! It needs to be in interval [0.0, 3.0).")

  if batch_size < 1:
    raise ValueError(
      "Value for 'batch_size' is invalid! It needs to be larger than zero.")

  if apply_sigmoid:
    if sigmoid_sensitivity is None:
      raise ValueError("Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!")
    if not 0.5 <= sigmoid_sensitivity <= 1.5:
      raise ValueError(
        "Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval [0.5, 1.5].")

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

  n_cpus = cpu_count()
  assert n_cpus is not None

  if custom_n_jobs is not None:
    if not 0 < custom_n_jobs <= n_cpus:
      raise ValueError(
        f"Value for 'custom_n_jobs' is invalid! It needs to be in interval [1, {n_cpus}].")

  if maxtasksperchild is not None and maxtasksperchild <= 0:
    raise ValueError(
      "Value for 'maxtasksperchild' is invalid! It needs to be None or larger than 0.")

  n_jobs = custom_n_jobs if custom_n_jobs is not None else n_cpus

  logger = getLogger(__name__)

  if len(audio_files) < n_jobs:
    adj_n_jobs = max(1, len(audio_files))
    n_jobs = adj_n_jobs
    logger.info("Lowered n_jobs according to file count.")

  logger.info(f"Using {n_jobs} CPU cores.")

  model: AudioModelV2M4TFLiteBase
  if custom_model is None:
    model = AudioModelV2M4TFLite(language="en_us", tflite_num_threads=1)
  else:
    if not isinstance(custom_model, AudioModelV2M4TFLiteBase):
      raise ValueError(
        "Value for 'custom_model' is invalid! It needs to be a subclass of 'AudioModelV2M4TFLiteBase'!")
    if custom_model.tflite_num_threads not in [0, 1]:
      raise ValueError(
        "Value for 'tflite_num_threads' in model is invalid! Multithreading in TFLite models is not supported. Please set tflite_num_threads to 1 to disable multithreading.")
    model = custom_model

  use_species_filter = species_filter is not None and len(species_filter) > 0
  if use_species_filter:
    assert species_filter is not None  # added for mypy
    species_filter_contains_unknown_species = not species_filter.issubset(model.species)
    if species_filter_contains_unknown_species:
      raise ValueError(
        f"At least one species defined in 'filter_species' is invalid! They need to be known species, e.g., {', '.join(model.species[:3])}")

  prediction_method = partial(
    predict_species_within_audio_file_core,
    min_confidence=min_confidence,
    batch_size=batch_size,
    chunk_overlap_s=chunk_overlap_s,
    use_bandpass=use_bandpass,
    bandpass_fmin=bandpass_fmin,
    bandpass_fmax=bandpass_fmax,
    apply_sigmoid=apply_sigmoid,
    sigmoid_sensitivity=sigmoid_sensitivity,
    species_filter=species_filter,
    silent=True,
  )

  process_prediction = partial(
    process_predict_species_within_audio_file,
    prediction_method=prediction_method,
  )

  errors: List[Tuple[Path, Exception]] = []

  with tqdm(total=len(audio_files), desc="Predicting species", unit="file", disable=silent) as pbar:
    with Pool(
      processes=n_jobs,
      initializer=init_pool,
      initargs=(model,),
      maxtasksperchild=maxtasksperchild,
    ) as pool:
      for audio_file, result in pool.imap_unordered(process_prediction, audio_files, chunksize=1):
        if isinstance(result, OrderedDict):
          yield audio_file, result
        else:
          assert isinstance(result, Exception)
          errors.append((audio_file, result))
        pbar.update()

  if len(errors) > 0:
    logger.error(f"{len(errors)}/{len(audio_files)} file(s) couldn't be processed!")

    for file, exception in errors:
      logger.exception(f"- {file.absolute()}", exc_info=exception)


process_model: Optional[AudioModelBaseV2M4] = None


def init_pool(model: AudioModelBaseV2M4) -> None:
  global process_model
  process_model = model


def process_predict_species_within_audio_file(audio_file: Path, prediction_method: Callable) -> Tuple[Path, Union[SpeciesPredictions, Exception]]:
  global process_model

  if not audio_file.is_file():
    return audio_file, ValueError(
      "Value for 'audio_file' is invalid! It needs to be a path to an existing audio file.")

  try:
    result = SpeciesPredictions(prediction_method(audio_file, model=process_model))
  except Exception as ex:
    return audio_file, ex

  return audio_file, result
