from pathlib import Path
from typing import final

from ordered_set import OrderedSet

from birdnet.base import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.geo_models.base import GeoModelBase


class GeoModelBaseV2_4(GeoModelBase):
  def __init__(self) -> None:
    super().__init__()
    self._model_path: Path | None = None
    self._species_list: OrderedSet[str] | None = None
    self._use_custom_model: bool | None = None

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @property
  def model_path(self) -> Path:
    assert self._model_path is not None
    return self._model_path

  @property
  def species_list(self) -> OrderedSet[str]:
    assert self._species_list is not None
    return self._species_list

  @classmethod
  @final
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return ACOUSTIC_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  def analyze(self) -> None:
    raise NotImplementedError("GeoModelBaseV2_4 does not implement analyze method.")
