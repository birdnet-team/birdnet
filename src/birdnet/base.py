from abc import ABC, abstractmethod
from typing import Literal

MODEL_TYPE_ACOUSTIC = "acoustic"
MODEL_TYPE_GEO = "geo"
MODEL_TYPES = Literal["acoustic", "geo"]

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


class ModelBase(ABC):
  def __init__(self) -> None:
    super().__init__()

  @classmethod
  @abstractmethod
  def get_backend(cls) -> MODEL_BACKENDS: ...
