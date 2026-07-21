"""Issue #55: Python 3.14 support.

TensorFlow does not yet publish wheels for Python 3.14, so on that interpreter
birdnet installs *without* TensorFlow. This module documents and pins the
TensorFlow-free surface:

- ``import birdnet`` works without TensorFlow installed.
- The acoustic 3.0 model runs via the ``onnx`` (and ``pt``) backend.
- Every TensorFlow-only path (``tf``/``pb`` backends, all geo models, the
  acoustic 2.4 and Perch models) fails with a clear, actionable ``ValueError``
  instead of a bare ``ModuleNotFoundError``.

The tests only run when TensorFlow is absent (i.e. on Python 3.14); they are
skipped on 3.11-3.13 where TensorFlow is installed.
"""

import pytest

import birdnet
from birdnet.core.backends import onnxruntime_installed, tf_installed

pytestmark = [
  pytest.mark.no_tf,
  pytest.mark.skipif(
    tf_installed(),
    reason="TensorFlow-free surface; runs only when TF is absent (Python 3.14).",
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
  with pytest.raises(ValueError, match="TensorFlow"):
    birdnet.load_perch_v2("CPU")


@pytest.mark.skipif(not onnxruntime_installed(), reason="onnxruntime not installed")
def test_acoustic_v3_onnx_predicts_without_tensorflow() -> None:
  model = birdnet.load("acoustic", "3.0", "onnx")
  predictions = model.predict("example/soundscape.wav")
  assert predictions is not None
