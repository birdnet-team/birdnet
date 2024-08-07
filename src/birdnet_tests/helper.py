
import math
import os
from itertools import count, islice
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm

from birdnet.types import Species, SpeciesPredictions, TimeInterval


def species_predictions_are_equal(
    actual: SpeciesPredictions,
    desired: SpeciesPredictions,
    precision: int
) -> bool:
  if actual.keys() != desired.keys():
    return False

  for interval in actual:
    actual_interval_predictions = actual[interval]
    desired_interval_predictions = desired[interval]

    if actual_interval_predictions.keys() != desired_interval_predictions.keys():
      return False

    for species in actual_interval_predictions:
      actual_score = actual_interval_predictions[species]
      desired_score = desired_interval_predictions[species]

      if not math.isclose(actual_score, desired_score, abs_tol=10**-precision):
        return False

  ordering_is_correct = True
  ordering_is_correct &= intervals_are_descending_order(actual)
  ordering_is_correct &= intervals_are_descending_order(desired)
  ordering_is_correct &= scores_are_descending_order(actual)
  ordering_is_correct &= scores_are_descending_order(desired)
  if not ordering_is_correct:
    return False

  return True


def scores_are_descending_order(predictions: SpeciesPredictions) -> bool:
  for _, interval_predictions in predictions.items():
    actual_values = list(interval_predictions.values())
    desired_values = sorted(actual_values, reverse=True)
    if actual_values != desired_values:
      return False

  return True


def intervals_are_descending_order(predictions: SpeciesPredictions) -> bool:
  actual_values = list(predictions.keys())
  desired_values = sorted(actual_values, reverse=False)
  result = actual_values == desired_values
  return result


def convert_predictions_to_numpy(predictions: SpeciesPredictions, /, *, custom_species: Optional[OrderedSet[Species]] = None) -> Tuple[npt.NDArray[np.float32], OrderedSet[TimeInterval], OrderedSet[Species]]:
  if custom_species is None:
    occurring_species = {
      species_name
      for k, p in predictions.items()
      for species_name in p.keys()
    }
    occurring_species_sorted = OrderedSet(sorted(occurring_species))
    output_species = occurring_species_sorted
  else:
    output_species = custom_species
  if len(output_species) == 0:
    raise NotImplementedError()
  result = np.zeros((len(output_species), len(predictions)), dtype=np.float32)
  for i, (_, pred) in enumerate(predictions.items()):
    for species, score in pred.items():
      key_index = output_species.index(species)
      result[key_index, i] = score
  y = output_species
  x = OrderedSet(predictions.keys())
  return result, x, y