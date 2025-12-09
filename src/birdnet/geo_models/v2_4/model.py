from __future__ import annotations

from pathlib import Path
from typing import Any, final

from ordered_set import OrderedSet

from birdnet.backends import (
  BackendLoader,
  VersionedGeoBackendProtocol,
)
from birdnet.geo_models.base import GeoModelBase
from birdnet.geo_models.inference.api import GeoPredictionSession
from birdnet.geo_models.inference.prediction_result import GeoPredictionResult
from birdnet.globals import (
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.helper import validate_species_list


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
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)
    self._backend_type = backend_type
    self._backend_custom_kwargs = backend_kwargs

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> GeoModelV2_4:
    result = GeoModelV2_4(
      model_path,
      species_list,
      use_custom_model=False,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> GeoModelV2_4:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list = validate_species_list(species_list)

    if check_validity:
      n_species_in_model = BackendLoader.check_model_can_be_loaded(
        model_path, backend_type, backend_kwargs
      )

      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, "
          f"but species list '{species_list.absolute()}' "
          f"has {len(loaded_species_list)} species!"
        )

    result = GeoModelV2_4(
      model_path,
      loaded_species_list,
      use_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  def predict_session(
    self,
    /,
    *,
    min_confidence: float = 0.03,
    half_precision: bool = False,
    device: str = "CPU",
  ) -> GeoPredictionSession:
    return GeoPredictionSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_is_custom=self.use_custom_model,
      model_version=self.get_version(),
      model_backend_type=self._backend_type,
      model_backend_custom_kwargs=self._backend_custom_kwargs,
      min_confidence=min_confidence,
      half_precision=half_precision,
      device=device,
    )

  def predict(
    self,
    latitude: float,
    longitude: float,
    /,
    *,
    week: int | None = None,
    min_confidence: float = 0.03,
    half_precision: bool = False,
    device: str = "CPU",
  ) -> GeoPredictionResult:
    with self.predict_session(
      min_confidence=min_confidence,
      half_precision=half_precision,
      device=device,
    ) as session:
      return session.run(
        latitude,
        longitude,
        week=week,
      )
