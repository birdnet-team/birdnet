from pathlib import Path
from typing import Set

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.types import Language, Species
from birdnet.utils import get_birdnet_app_data_folder

AVAILABLE_LANGUAGES: Set[Language] = {
    "sv", "da", "hu", "th", "pt", "fr", "cs", "af", "en_uk", "uk", "it", "ja", "sl", "pl", "ko", "es", "de", "tr", "ru", "en_us", "no", "sk", "ar", "fi", "ro", "nl", "zh"
}


class ModelBaseV2M4():
  def __init__(self, species_list: OrderedSet[Species]) -> None:
    self._species_list = species_list

  @property
  def species(self) -> OrderedSet[Species]:
    return self._species_list


class MetaModelBaseV2M4(ModelBaseV2M4):
  def __init__(self, species_list: OrderedSet[Species]) -> None:
    super().__init__(species_list)

  def predict_species(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError()


class AudioModelBaseV2M4(ModelBaseV2M4):
  def __init__(self, species_list: OrderedSet[Species]) -> None:
    super().__init__(species_list)

    self._sig_fmin: int = 0
    self._sig_fmax: int = 15_000
    self._sample_rate: int = 48_000
    self._chunk_size_s: float = 3.0

  @property
  def sig_fmin(self) -> int:
    return self._sig_fmin

  @property
  def sig_fmax(self) -> int:
    return self._sig_fmax

  @property
  def sample_rate(self) -> int:
    return self._sample_rate

  @property
  def chunk_size_s(self) -> float:
    return self._chunk_size_s

  def predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError()


def validate_language(language: Language):
  if language not in AVAILABLE_LANGUAGES:
    raise ValueError(
      f"Language '{language}' is not available! Choose from: {', '.join(sorted(AVAILABLE_LANGUAGES))}.")


def get_internal_version_app_data_folder() -> Path:
  birdnet_app_data = get_birdnet_app_data_folder()
  model_version_folder = birdnet_app_data / "models" / "v2.4"
  return model_version_folder
