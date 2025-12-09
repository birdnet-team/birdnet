import logging

from birdnet.acoustic_models.inference.encoding.result import (
  AcousticDataEncodingResult,
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet.acoustic_models.inference.perf_tracker import AcousticProgressStats
from birdnet.acoustic_models.inference.prediction.result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
  AcousticPredictionResultBase,
)
from birdnet.acoustic_models.inference_pipeline.api import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet.acoustic_models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.geo_models.inference.api import GeoPredictionSession
from birdnet.geo_models.inference.prediction_result import GeoPredictionResult
from birdnet.geo_models.v2_4.model import GeoModelV2_4
from birdnet.logging_utils import get_package_logger, init_package_logger
from birdnet.model_loader import load, load_custom, load_perch_v2  # noqa: F401

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
