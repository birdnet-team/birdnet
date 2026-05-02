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
)
from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.model_loader import load, load_custom, load_perch_v2  # noqa: F401
from birdnet.utils.logging_utils import get_package_logger, init_package_logger

init_package_logger(logging.INFO)
__all__ = [
  # logging
  "get_package_logger",
  # model loading
  "load",
  "load_custom",
  "load_perch_v2",
  # acoustic encoding
  "AcousticEncodingSession",
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
  "AcousticModelPerchV2",
  # geo prediction
  "GeoPredictionSession",
  "GeoPredictionResult",
  # geo models
  "GeoModelV2_4",
]
