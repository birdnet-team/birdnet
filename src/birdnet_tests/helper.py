
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.types import Species, SpeciesPrediction, SpeciesPredictions, TimeInterval

TEST_RESULTS_DIR = Path("src/birdnet_tests/test_results")
TEST_FILES_DIR = Path("src/birdnet_tests/test_files")
# Duration: 120s
TEST_FILE_WAV = TEST_FILES_DIR / "soundscape.wav"


def species_prediction_is_equal(
    actual: SpeciesPrediction,
    desired: SpeciesPrediction,
    decimal: int
) -> bool:
  if actual.keys() != desired.keys():
    return False

  for species in actual:
    actual_score = actual[species]
    desired_score = desired[species]

    if not math.isclose(actual_score, desired_score, abs_tol=10**-decimal):
      return False

  ordering_is_correct = True
  ordering_is_correct &= species_scores_are_descending_order(actual)
  ordering_is_correct &= species_scores_are_descending_order(desired)
  if not ordering_is_correct:
    return False

  return True


def species_predictions_are_equal(
    actual: SpeciesPredictions,
    desired: SpeciesPredictions,
    decimal: int
) -> bool:
  if actual.keys() != desired.keys():
    return False

  for interval in actual:
    actual_interval_predictions = actual[interval]
    desired_interval_predictions = desired[interval]

    if not species_prediction_is_equal(actual_interval_predictions, desired_interval_predictions, decimal):
      return False

  ordering_is_correct = True
  ordering_is_correct &= intervals_are_descending_order(actual)
  ordering_is_correct &= intervals_are_descending_order(desired)
  if not ordering_is_correct:
    return False

  return True


def species_scores_are_descending_order(predictions: SpeciesPrediction) -> bool:
  actual_values = list(predictions.items())
  desired_values = sorted(actual_values, key=lambda kv: (kv[1] * -1, kv[0]), reverse=False)
  return actual_values == desired_values


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
