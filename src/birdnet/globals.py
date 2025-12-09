from typing import Literal

import numpy as np
import numpy.typing as npt

NA = "N/A"

LIBRARY_TFLITE = "tflite"  # default: TFLite Interpreter
LIBRARY_LITERT = "litert"  # LiteRT Interpreter
LIBRARY_TYPES = Literal["tflite", "litert"]
VALID_LIBRARY_TYPES = [
  LIBRARY_TFLITE,
  LIBRARY_LITERT,
]

# name of the parameter to specify the library when loading a TF model
LIBRARY_TF_PARAM = "library"
LIBRARY_TF_DEFAULT = LIBRARY_TFLITE

# name of the parameter to specify whether a custom PB model is a Raven model
CUSTOM_PB_IS_RAVEN_PARAM = "is_raven"
CUSTOM_PB_IS_RAVEN_DEFAULT = True

MODEL_FAMILY_BIRDNET = "birdnet"
MODEL_FAMILY_PERCH = "perch"
MODEL_FAMILIES = Literal["birdnet", "perch"]
VALID_MODEL_FAMILIES = [
  MODEL_FAMILY_BIRDNET,
  MODEL_FAMILY_PERCH,
]

MODEL_TYPE_ACOUSTIC = "acoustic"
MODEL_TYPE_GEO = "geo"
MODEL_TYPES = Literal["acoustic", "geo"]
VALID_MODEL_TYPES = [
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPE_GEO,
]

ACOUSTIC_MODEL_VERSION_V2_4 = "2.4"
ACOUSTIC_MODEL_VERSION_V2 = "2"
ACOUSTIC_MODEL_VERSIONS = Literal["2.4",]
VALID_ACOUSTIC_MODEL_VERSIONS = [ACOUSTIC_MODEL_VERSION_V2_4, ACOUSTIC_MODEL_VERSION_V2]

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
]  # TODO: use `from typing import get_args`
MODEL_LANGUAGE_EN_US = "en_us"
MODEL_LANGUAGES = (
  Literal[
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
  | Literal["latin"]
)
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
] + ["latin"]


PKG_NAME = "birdnet"

# flag for "can be written to" = free
WRITABLE_FLAG = np.uint8(0)

# flag for "currently being written to"
WRITING_FLAG = np.uint8(1)

# flag for "can be read from" = preloaded
READABLE_FLAG = np.uint8(2)

# flag for "busy", i.e., currently being processed
READING_FLAG = np.uint8(3)

IntArray = npt.NDArray[np.integer]
FloatArray = npt.NDArray[np.floating]

Float32Array = npt.NDArray[np.float32]
