"""Names that BirdNET-Analyzer imports from this package.

None of them is read inside the package, so a dead-code sweep sees them as
unused; removing or renaming one breaks the Analyzer on import (issue #118).
The list mirrors the Analyzer's imports at its commit c2c71a1.
"""

import importlib

import pytest

ANALYZER_IMPORTS = [
  ("birdnet.acoustic.inference.configs", "InferenceConfig"),
  (
    "birdnet.acoustic.inference.core.encoding.encoding_result",
    "AcousticFileEncodingResult",
  ),
  ("birdnet.acoustic.inference.core.perf_tracker", "AcousticProgressStats"),
  (
    "birdnet.acoustic.inference.core.prediction.prediction_result",
    "AcousticFilePredictionResult",
  ),
  ("birdnet.acoustic.inference.session", "AcousticEncodingSession"),
  ("birdnet.acoustic.inference.session", "AcousticSessionBase"),
  ("birdnet.acoustic.models.v2_4.tf", "AcousticTFDownloaderV2_4"),
  ("birdnet.acoustic.models.v3_0.onnx", "AcousticOnnxDownloaderV3_0"),
  ("birdnet.globals", "ACOUSTIC_MODEL_VERSIONS"),
  ("birdnet.globals", "GEO_MODEL_VERSIONS"),
  ("birdnet.globals", "MODEL_LANGUAGES"),
  ("birdnet.globals", "MODEL_LANGUAGES_V2_4"),
  ("birdnet.globals", "MODEL_LANGUAGES_V3_0"),
  ("birdnet.globals", "MODEL_LANGUAGE_EN_US"),
  ("birdnet.globals", "MODEL_PRECISION_FP32"),
  ("birdnet.globals", "VALID_MODEL_LANGUAGES"),
  ("birdnet.globals", "VALID_MODEL_LANGUAGES_V2_4"),
  ("birdnet.globals", "VALID_MODEL_LANGUAGES_V3_0"),
]


@pytest.mark.no_tf
@pytest.mark.parametrize(("module", "name"), ANALYZER_IMPORTS)
def test_name_the_analyzer_imports_exists(module: str, name: str) -> None:
  assert hasattr(importlib.import_module(module), name)
