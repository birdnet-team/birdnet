from operator import itemgetter
from typing import Optional

import numpy as np

from birdnet.models.v2m4.model_v2m4_base import MetaModelBaseV2M4
from birdnet.models.v2m4.model_v2m4_protobuf import MetaModelV2M4Protobuf
from birdnet.types import SpeciesPrediction


def predict_species_at_location_and_time(
    latitude: float,
    longitude: float,
    /,
    *,
    week: Optional[int] = None,
    min_confidence: float = 0.03,
    custom_model: Optional[MetaModelBaseV2M4] = None,
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

  if week is None:
    week = -1
  assert week is not None

  model: MetaModelBaseV2M4
  if custom_model is None:
    model = MetaModelV2M4Protobuf()
  else:
    model = custom_model

  sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

  l_filter = model.predict_species(sample)

  prediction = (
    (species, score)
    for species, score in zip(model.species, l_filter)
    if score >= min_confidence
  )

  sorted_prediction = SpeciesPrediction(
    sorted(prediction, key=itemgetter(1), reverse=True)
  )

  return sorted_prediction
