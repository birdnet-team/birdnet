from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from ordered_set import OrderedSet

LIBRARY_TF = "tf"
LIBRARY_LITERT = "litert"
LIBRARY_TYPES = Literal["tf", "litert"]
VALID_LIBRARY_TYPES = [
  LIBRARY_TF,
  LIBRARY_LITERT,
]

MODEL_TYPE_ACOUSTIC = "acoustic"
MODEL_TYPE_GEO = "geo"
MODEL_TYPES = Literal["acoustic", "geo"]
VALID_MODEL_TYPES = [
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPE_GEO,
]

ACOUSTIC_MODEL_VERSION_V2_4 = "2.4"
ACOUSTIC_MODEL_VERSIONS = Literal["2.4",]
VALID_ACOUSTIC_MODEL_VERSIONS = [
  ACOUSTIC_MODEL_VERSION_V2_4,
]

GEO_MODEL_VERSION_V2_4 = "2.4"
GEO_MODEL_VERSIONS = Literal["2.4",]
VALID_GEO_MODEL_VERSIONS = [
  GEO_MODEL_VERSION_V2_4,
]

MODEL_BACKEND_TF = "tf"
MODEL_BACKEND_PB = "pb"
MODEL_BACKENDS = Literal["tf", "pb"]
VALID_MODEL_BACKENDS = [
  MODEL_BACKEND_TF,
  MODEL_BACKEND_PB,
]

MODEL_PRECISION_INT8 = "int8"
MODEL_PRECISION_FLOAT16 = "fp16"
MODEL_PRECISION_FLOAT32 = "fp32"
MODEL_PRECISIONS = Literal["int8", "fp16", "fp32"]
VALID_MODEL_PRECISIONS = [
  MODEL_PRECISION_INT8,
  MODEL_PRECISION_FLOAT16,
  MODEL_PRECISION_FLOAT32,
]
MODEL_LANGUAGE_EN_US = "en_us"
MODEL_LANGUAGES = Literal[
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
]
VALID_MODEL_LANGUAGES = [
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
]


class ModelBase(ABC):
  def __init__(
    self, model_path: Path, species_list: OrderedSet[str], use_custom_model: bool
  ) -> None:
    super().__init__()
    self._model_path = model_path
    self._species_list = species_list
    self._use_custom_model = use_custom_model

  @classmethod
  @abstractmethod
  def get_backend(cls) -> MODEL_BACKENDS: ...

  @property
  def model_path(self) -> Path:
    return self._model_path

  @property
  def species_list(self) -> OrderedSet[str]:
    return self._species_list

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @property
  def use_custom_model(self) -> bool:
    return self._use_custom_model
