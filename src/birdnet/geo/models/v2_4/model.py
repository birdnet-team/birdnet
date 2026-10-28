from __future__ import annotations

from typing import final

from ordered_set import OrderedSet

from birdnet.geo.models.base import GeoModelBase
from birdnet.globals import (
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)

_YEAR_ROUND_WEEK_INPUTS = (-1.0,)


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


class GeoModelV2_4(GeoModelBase):
  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  @classmethod
  @final
  def get_n_classes(cls) -> int:
    """Return the 6,522 classes of the official 2.4 output.

    The label list of the acoustic 2.4 model, so it includes that model's 11
    classes that are not species, such as Dog, Engine and Noise.
    """
    return 6522
