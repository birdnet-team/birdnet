"""The names ``birdnet`` exports: the contract with every consumer.

A removal here is a breaking change, whether or not anything inside the package
reads the name (issue #118).
"""

import pytest

import birdnet

pytestmark = pytest.mark.no_tf

EXPORTED = [
  "ACOUSTIC_MODEL_VERSIONS",
  "AcousticDataEncodingResult",
  "AcousticDataPredictionResult",
  "AcousticEncodingResultBase",
  "AcousticEncodingSession",
  "AcousticFileEncodingResult",
  "AcousticFilePredictionResult",
  "AcousticModelPerchV2",
  "AcousticModelV2_4",
  "AcousticModelV3_0",
  "AcousticPredictionResultBase",
  "AcousticPredictionSession",
  "AcousticProgressStats",
  "AcousticSessionBase",
  "CUSTOM_CLASSIFIER_TYPES",
  "CustomTFLiteInfo",
  "DownloadProgress",
  "DownloadProgressCallback",
  "DownloadStatus",
  "GEO_MODEL_VERSIONS",
  "GeoModelV2_4",
  "GeoModelV3_0",
  "GeoPredictionResult",
  "GeoPredictionSession",
  "LIBRARY_TYPES",
  "MODEL_BACKENDS",
  "MODEL_LANGUAGES",
  "MODEL_LANGUAGES_V2_4",
  "MODEL_LANGUAGES_V3_0",
  "MODEL_LANGUAGE_EN_US",
  "MODEL_PRECISIONS",
  "MODEL_PRECISION_FP32",
  "MODEL_TYPES",
  "VALID_CUSTOM_CLASSIFIER_TYPES",
  "VALID_LIBRARY_TYPES",
  "VALID_MODEL_BACKENDS",
  "VALID_MODEL_LANGUAGES",
  "VALID_MODEL_LANGUAGES_V2_4",
  "VALID_MODEL_LANGUAGES_V3_0",
  "VALID_MODEL_PRECISIONS",
  "VALID_MODEL_TYPES",
  "download_progress_callback",
  "get_download_progress_callback",
  "get_package_logger",
  "inspect_custom_tflite",
  "load",
  "load_custom",
  "load_perch_v2",
  "set_download_progress_callback",
]


def test_exports_match_the_contract() -> None:
  assert sorted(birdnet.__all__) == EXPORTED


def test_every_export_resolves() -> None:
  for name in EXPORTED:
    assert hasattr(birdnet, name), name
