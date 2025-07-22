from pathlib import Path
from typing import final

from ordered_set import OrderedSet

from birdnet.base import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.geo_models.base import GeoModelBase


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

  def analyze(self) -> None:
    raise NotImplementedError("GeoModelBaseV2_4 does not implement analyze method.")
