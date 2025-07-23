from pathlib import Path
from typing import final

import numpy as np
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

from birdnet.geo_models.base import GeoModelBase
from birdnet.geo_models.inference.prediction_result import PredictionResult
from birdnet.globals import (
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.helper import uint_dtype_for


class GeoDownloaderBaseV2_4:
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(
    (
      "af",
      "ar",
      "cs",
      "da",
      "de",
      "en_uk",
      "en_us",
      "es",
      "fi",
      "fr",
      "hu",
      "it",
      "ja",
      "ko",
      "nl",
      "no",
      "pl",
      "pt",
      "ro",
      "ru",
      "sk",
      "sl",
      "sv",
      "th",
      "tr",
      "uk",
      "zh",
    )
  )


class GeoModelBaseV2_4(GeoModelBase):
  def __init__(
    self, model_path: Path, species_list: OrderedSet[str], use_custom_model: bool
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)

  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  def _predict(
    self,
    latitude: float,
    longitude: float,
    backend_kwargs: dict,
    /,
    *,
    week: int | None = None,
    min_confidence: float = 0.03,
    device: str = "CPU",
    half_precision: bool = True,
  ) -> PredictionResult:
    if not -90 <= latitude <= 90:
      raise ValueError(
        "Value for 'latitude' is invalid! It needs to be in interval [-90, 90]."
      )

    if not -180 <= longitude <= 180:
      raise ValueError(
        "Value for 'longitude' is invalid! It needs to be in interval [-180, 180]."
      )

    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0)."
      )

    if week is not None and not (1 <= week <= 48):
      raise ValueError(
        "Value for 'week' is invalid! It needs to be either None or in interval [1, 48]."
      )

    if week is None:
      week = -1
    assert week is not None

    sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

    backend_type = self.get_backend_type()

    try:
      backend = backend_type(**backend_kwargs)
      backend.load()
    except Exception as exc:
      raise ValueError("Failed to load backend.") from exc

    prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

    res = backend.infer(sample, device_name=device)
    assert res.dtype == np.float32
    res = res.astype(prob_dtype, copy=False)

    res = np.squeeze(res, axis=0)

    species_ids = np.arange(
      len(self.species_list),
      dtype=uint_dtype_for(
        max(0, len(self.species_list) - 1),
      ),
    )

    invalid_mask = res < min_confidence
    prediction = PredictionResult(
      species_list=self.species_list,
      species_probs=res,
      species_ids=species_ids,
      species_masked=invalid_mask,
    )

    return prediction
