from typing import Final, Literal, get_args

import numpy as np
import numpy.typing as npt

NA: Final = "N/A"

LIBRARY_TFLITE: Final = "tflite"  # default: TFLite Interpreter
LIBRARY_LITERT: Final = "litert"  # LiteRT Interpreter
LIBRARY_TYPES = Literal["tflite", "litert"]
VALID_LIBRARY_TYPES = list(get_args(LIBRARY_TYPES))

# name of the parameter to specify the library when loading a TF model
LIBRARY_TF_PARAM: Final = "library"
LIBRARY_TF_DEFAULT: Final = LIBRARY_TFLITE

# name of the parameter to specify whether a custom PB model is a Raven model
CUSTOM_PB_IS_RAVEN_PARAM: Final = "is_raven"
CUSTOM_PB_IS_RAVEN_DEFAULT = True
# 3.0 custom SavedModels loaded as the official layout before Raven support
# existed for them, so the default there stays False.
CUSTOM_PB_IS_RAVEN_DEFAULT_V3_0 = False

CUSTOM_CLASSIFIER_REPLACE: Final = "replace"
CUSTOM_CLASSIFIER_APPEND: Final = "append"
CUSTOM_CLASSIFIER_REPLACE_HIDDEN: Final = "replace_hidden"
CUSTOM_CLASSIFIER_APPEND_HIDDEN: Final = "append_hidden"
CUSTOM_CLASSIFIER_TYPES = Literal[
  "replace", "append", "replace_hidden", "append_hidden"
]
VALID_CUSTOM_CLASSIFIER_TYPES = list(get_args(CUSTOM_CLASSIFIER_TYPES))
# name of the parameter to specify the classifier type for custom TFLite models
CUSTOM_CLASSIFIER_TF_TYPE_PARAM: Final = "classifier_type"

MODEL_TYPE_ACOUSTIC: Final = "acoustic"
MODEL_TYPE_GEO: Final = "geo"
MODEL_TYPES = Literal["acoustic", "geo"]
VALID_MODEL_TYPES = list(get_args(MODEL_TYPES))

ACOUSTIC_MODEL_VERSION_V2_4: Final = "2.4"
ACOUSTIC_MODEL_VERSION_V3_0: Final = "3.0"
ACOUSTIC_MODEL_VERSIONS = Literal["2.4", "3.0"]
VALID_ACOUSTIC_MODEL_VERSIONS = list(get_args(ACOUSTIC_MODEL_VERSIONS))
# load_custom only: detect the version of a custom TFLite classifier from the file
ACOUSTIC_MODEL_VERSION_AUTO: Final = "auto"

GEO_MODEL_VERSION_V2_4: Final = "2.4"
GEO_MODEL_VERSION_V3_0: Final = "3.0"
GEO_MODEL_VERSIONS = Literal["2.4", "3.0"]
VALID_GEO_MODEL_VERSIONS = list(get_args(GEO_MODEL_VERSIONS))

GEO_YEAR_ROUND_AGGREGATION_MAX: Final = "max"
GEO_YEAR_ROUND_AGGREGATION_AVERAGE: Final = "average"
GEO_YEAR_ROUND_AGGREGATIONS = Literal["max", "average"]
VALID_GEO_YEAR_ROUND_AGGREGATIONS = list(get_args(GEO_YEAR_ROUND_AGGREGATIONS))

MODEL_BACKEND_TF: Final = "tf"
MODEL_BACKEND_PB: Final = "pb"
MODEL_BACKEND_PT: Final = "pt"
MODEL_BACKEND_ONNX: Final = "onnx"
MODEL_BACKENDS = Literal["tf", "pb", "pt", "onnx"]
VALID_MODEL_BACKENDS = list(get_args(MODEL_BACKENDS))

MODEL_PRECISION_INT8: Final = "int8"
MODEL_PRECISION_FP16: Final = "fp16"
MODEL_PRECISION_FP32: Final = "fp32"
MODEL_PRECISIONS = Literal["int8", "fp16", "fp32"]
VALID_MODEL_PRECISIONS = list(get_args(MODEL_PRECISIONS))
MODEL_LANGUAGE_EN_US: Final = "en_us"
MODEL_LANGUAGES_V2_4 = Literal[
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
VALID_MODEL_LANGUAGES_V2_4 = list(get_args(MODEL_LANGUAGES_V2_4))

MODEL_LANGUAGES_V3_0 = Literal[
  "bg",
  "ca",
  "cs",
  "cy",
  "da",
  "de",
  "en_us",
  "es",
  "es_ec",
  "es_es",
  "es_mx",
  "fa",
  "fi",
  "fr",
  "hr",
  "ja",
  "lt",
  "nl",
  "no",
  "pl",
  "pt",
  "pt_pt",
  "ru",
  "sk",
  "sr",
  "sv",
  "tr",
  "uk",
  "zh",
]
VALID_MODEL_LANGUAGES_V3_0 = list(get_args(MODEL_LANGUAGES_V3_0))

# Downstream packages import these; nothing in this package reads them.
MODEL_LANGUAGES = MODEL_LANGUAGES_V2_4 | MODEL_LANGUAGES_V3_0
VALID_MODEL_LANGUAGES = VALID_MODEL_LANGUAGES_V2_4 + VALID_MODEL_LANGUAGES_V3_0

PKG_NAME: Final = "birdnet"

ENV_VAR_APP_DATA: Final = "BIRDNET_APP_DATA"

ENV_VAR_START_METHOD: Final = "BIRDNET_START_METHOD"

# Set to anything but "0" to let TensorFlow's native startup output through.
ENV_VAR_TF_VERBOSE: Final = "BIRDNET_TF_VERBOSE"

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
