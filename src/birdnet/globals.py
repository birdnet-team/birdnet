from typing import Literal

import numpy as np

LIBRARY_TF = "tf"  # default
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
MODEL_PRECISION_FP16 = "fp16"
MODEL_PRECISION_FP32 = "fp32"
MODEL_PRECISIONS = Literal["int8", "fp16", "fp32"]
VALID_MODEL_PRECISIONS = [
  MODEL_PRECISION_INT8,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
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

PKG_NAME = "birdnet"

# flag for "can be written to" = free
WRITABLE_FLAG = np.uint8(0)

# flag for "currently being written to"
WRITING_FLAG = np.uint8(1)

# flag for "can be read from" = preloaded
READABLE_FLAG = np.uint8(2)

# flag for "busy", i.e., currently being processed
READING_FLAG = np.uint8(3)
