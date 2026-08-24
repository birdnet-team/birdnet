"""Issue #55: the TensorFlow-free surface.

TensorFlow is an optional dependency (``birdnet[tf]``) and has no wheels for
Python 3.14, so a base install runs without it. This module documents and pins
what works there:

- ``import birdnet`` works without TensorFlow installed.
- The acoustic 3.0 and geo 3.0 models run via the ``onnx`` and ``pt`` backends.
- The ``tf`` backend runs the .tflite models on ai-edge-litert with
  ``library="litert"`` (acoustic 2.4, geo 2.4 and custom 2.4 classifiers) --
  ai-edge-litert never imports TensorFlow, and this lane, not a mocked
  ``tf_installed``, is what proves that import graph is clean.
- Every TensorFlow-only path (``tf`` with the default ``tflite`` interpreter,
  ``pb``, and the Perch model) fails with a clear, actionable ``ValueError``
  instead of a bare ``ModuleNotFoundError``.

The tests only run when TensorFlow is absent (the py314 and py313-notf lanes);
they are skipped where TensorFlow is installed.
"""

from pathlib import Path

import pytest

import birdnet
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32CustomAppendV2_4
from birdnet.core.backends import (
  litert_installed,
  onnxruntime_installed,
  tf_installed,
  torch_installed,
)
from birdnet_tests.helper import ensure_not_intel_macos_or_skip
from birdnet_tests.test_files import TEST_FILES_DIR

pytestmark = [
  pytest.mark.no_tf,
  pytest.mark.skipif(
    tf_installed(),
    reason="TensorFlow-free surface; runs only when TensorFlow is not installed.",
  ),
]


def test_import_birdnet_without_tensorflow() -> None:
  assert not tf_installed()
  assert birdnet is not None


@pytest.mark.parametrize(
  ("model_type", "version", "backend"),
  [
    ("acoustic", "2.4", "tf"),
    ("acoustic", "2.4", "pb"),
    ("acoustic", "3.0", "tf"),
    ("acoustic", "3.0", "pb"),
    ("geo", "2.4", "tf"),
    ("geo", "2.4", "pb"),
    ("geo", "3.0", "tf"),
    ("geo", "3.0", "pb"),
  ],
)
def test_tf_backends_raise_clear_error_without_tensorflow(
  model_type: str, version: str, backend: str
) -> None:
  with pytest.raises(ValueError, match="TensorFlow"):
    birdnet.load(model_type, version, backend)


def test_perch_raises_clear_error_without_tensorflow() -> None:
  ensure_not_intel_macos_or_skip()  # rejected there before any TensorFlow check
  with pytest.raises(ValueError, match="TensorFlow"):
    birdnet.load_perch_v2("CPU")


@pytest.mark.parametrize("backend", ["tf", "pb"])
def test_load_custom_tf_backends_raise_clear_error_without_tensorflow(
  backend: str, tmp_path: Path
) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(ValueError, match="TensorFlow"):
    birdnet.load_custom("acoustic", "2.4", backend, tmp_path, species)


@pytest.mark.skipif(not onnxruntime_installed(), reason="onnxruntime not installed")
def test_acoustic_v3_onnx_predicts_without_tensorflow() -> None:
  model = birdnet.load("acoustic", "3.0", "onnx")
  predictions = model.predict("example/soundscape.wav")
  assert predictions is not None


@pytest.mark.skipif(not onnxruntime_installed(), reason="onnxruntime not installed")
def test_geo_v3_onnx_predicts_without_tensorflow() -> None:
  model = birdnet.load("geo", "3.0", "onnx")
  predictions = model.predict(42.5, -76.45, week=4)
  assert predictions is not None


@pytest.mark.skipif(not torch_installed(), reason="torch not installed")
def test_geo_v3_pt_predicts_without_tensorflow() -> None:
  model = birdnet.load("geo", "3.0", "pt")
  predictions = model.predict(42.5, -76.45, week=4)
  assert predictions is not None


@pytest.mark.skipif(not torch_installed(), reason="torch not installed")
def test_acoustic_v3_pt_predicts_without_tensorflow() -> None:
  model = birdnet.load("acoustic", "3.0", "pt")
  predictions = model.predict("example/soundscape.wav")
  assert predictions is not None


# ----------------------------- tf backend on ai-edge-litert -----------------------
# Each 2.4 model is loaded by exactly one test: the TensorFlow-free lanes have no
# serialized `load_model` phase, and the 2.4 zip extraction (unlike the single-file
# 3.0 downloads and the label writes) is not atomic, so two tests fetching the same
# model concurrently on a cold cache would race.

_needs_litert = pytest.mark.skipif(
  not litert_installed(), reason="ai-edge-litert not installed"
)


@pytest.mark.litert
@_needs_litert
def test_tf_backend_error_points_at_litert_without_tensorflow() -> None:
  with pytest.raises(ValueError, match=r"pass library='litert'"):
    birdnet.load("acoustic", "2.4", "tf")


@pytest.mark.litert
@_needs_litert
def test_acoustic_v2_4_tf_litert_predicts_and_encodes_without_tensorflow() -> None:
  model = birdnet.load("acoustic", "2.4", "tf", library="litert")
  predictions = model.predict("example/soundscape.wav")
  # 120 s soundscape -> 40 segments, default top_k=5
  assert predictions.species_probs.shape == (1, 40, 5)
  embeddings = model.encode("example/soundscape.wav")
  assert embeddings.embeddings.shape == (1, 40, 1024)


@pytest.mark.litert
@_needs_litert
def test_geo_v2_4_tf_litert_predicts_without_tensorflow() -> None:
  model = birdnet.load("geo", "2.4", "tf", library="litert")
  predictions = model.predict(42.5, -76.45, week=4)
  assert predictions.species_probs.shape == (6522,)


@pytest.mark.litert
@_needs_litert
def test_custom_acoustic_v2_4_tf_litert_detects_classifier_without_tensorflow() -> None:
  # check_validity=True runs the classifier-type detection, i.e. opens the
  # .tflite file with the litert interpreter in a subprocess.
  model = birdnet.load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/append.tflite",
    TEST_FILES_DIR / "custom_models/tf/append_Labels.txt",
    library="litert",
    check_validity=True,
  )
  assert model.backend_type is AcousticTFBackendFP32CustomAppendV2_4
