"""The cached file name must carry the declared checksum's prefix.

Without it, two installed releases sharing one app data directory judge each
other's download stale and overwrite it, and a model swapped under the same
family version with an unchanged byte size is never re-fetched.
"""

import pytest

from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0
from birdnet.acoustic.models.v3_0.onnx import models as onnx_models
from birdnet.acoustic.models.v3_0.pt import AcousticPTDownloaderV3_0
from birdnet.acoustic.models.v3_0.pt import models as pt_models
from birdnet.acoustic.models.v3_0.tf import AcousticTFDownloaderV3_0
from birdnet.acoustic.models.v3_0.tf import models as tf_models
from birdnet.globals import MODEL_PRECISION_FP32


@pytest.mark.no_tf
def test_tf_model_paths() -> None:
  for precision, info in tf_models.items():
    model_path, _ = AcousticTFDownloaderV3_0._get_paths(precision)
    assert model_path.name == f"model-{precision}-{info.content_tag}.tflite"


@pytest.mark.no_tf
def test_onnx_model_paths() -> None:
  for precision, info in onnx_models.items():
    model_path, _ = AcousticOnnxDownloaderV3_0._get_paths(precision)
    assert model_path.name == f"model-{precision}-{info.content_tag}.onnx"


@pytest.mark.no_tf
def test_pt_model_path() -> None:
  model_path, _ = AcousticPTDownloaderV3_0._get_paths()
  info = pt_models[MODEL_PRECISION_FP32]
  assert model_path.name == f"model-fp32-{info.content_tag}.pt"
