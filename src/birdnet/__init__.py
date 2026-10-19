import logging

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticDataEncodingResult,
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
  AcousticPredictionResultBase,
)
from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
  AcousticSessionBase,
)
from birdnet.acoustic.models.custom_tflite import CustomTFLiteInfo
from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  CUSTOM_CLASSIFIER_TYPES,
  GEO_MODEL_VERSIONS,
  LIBRARY_TF_DEFAULT,
  LIBRARY_TYPES,
  MODEL_BACKENDS,
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES,
  MODEL_LANGUAGES_V2_4,
  MODEL_LANGUAGES_V3_0,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
  MODEL_TYPES,
  VALID_ACOUSTIC_MODEL_VERSIONS,
  VALID_CUSTOM_CLASSIFIER_TYPES,
  VALID_GEO_MODEL_VERSIONS,
  VALID_LIBRARY_TYPES,
  VALID_MODEL_BACKENDS,
  VALID_MODEL_LANGUAGES,
  VALID_MODEL_LANGUAGES_V2_4,
  VALID_MODEL_LANGUAGES_V3_0,
  VALID_MODEL_PRECISIONS,
  VALID_MODEL_TYPES,
)
from birdnet.model_loader import (
  inspect_custom_tflite,
  load,
  load_custom,
  load_perch_v2,
)
from birdnet.utils.download_progress import (
  DownloadProgress,
  DownloadProgressCallback,
  DownloadStatus,
  download_progress_callback,
  get_download_progress_callback,
  set_download_progress_callback,
)
from birdnet.utils.logging_utils import get_package_logger, init_package_logger

init_package_logger(logging.INFO)
__all__ = [
  # logging
  "get_package_logger",
  # model loading
  "load",
  "load_custom",
  "load_perch_v2",
  "inspect_custom_tflite",
  "CustomTFLiteInfo",
  # download progress
  "DownloadProgress",
  "DownloadProgressCallback",
  "DownloadStatus",
  "download_progress_callback",
  "get_download_progress_callback",
  "set_download_progress_callback",
  # acoustic encoding
  "AcousticEncodingSession",
  "AcousticSessionBase",
  "AcousticEncodingResultBase",
  "AcousticDataEncodingResult",
  "AcousticFileEncodingResult",
  # acoustic prediction
  "AcousticPredictionSession",
  "AcousticPredictionResultBase",
  "AcousticDataPredictionResult",
  "AcousticFilePredictionResult",
  # performance tracking
  "AcousticProgressStats",
  # acoustic models
  "AcousticModelV2_4",
  "AcousticModelV3_0",
  "AcousticModelPerchV2",
  # geo prediction
  "GeoPredictionSession",
  "GeoPredictionResult",
  # geo models
  "GeoModelV2_4",
  "GeoModelV3_0",
  # model vocabulary: versions, backends, precisions, languages
  "ACOUSTIC_MODEL_VERSIONS",
  "CUSTOM_CLASSIFIER_TYPES",
  "GEO_MODEL_VERSIONS",
  "LIBRARY_TF_DEFAULT",
  "LIBRARY_TYPES",
  "MODEL_BACKENDS",
  "MODEL_LANGUAGE_EN_US",
  "MODEL_LANGUAGES",
  "MODEL_LANGUAGES_V2_4",
  "MODEL_LANGUAGES_V3_0",
  "MODEL_PRECISION_FP32",
  "MODEL_PRECISIONS",
  "MODEL_TYPES",
  "VALID_ACOUSTIC_MODEL_VERSIONS",
  "VALID_CUSTOM_CLASSIFIER_TYPES",
  "VALID_GEO_MODEL_VERSIONS",
  "VALID_LIBRARY_TYPES",
  "VALID_MODEL_BACKENDS",
  "VALID_MODEL_LANGUAGES",
  "VALID_MODEL_LANGUAGES_V2_4",
  "VALID_MODEL_LANGUAGES_V3_0",
  "VALID_MODEL_PRECISIONS",
  "VALID_MODEL_TYPES",
]
