"""Issue #110: the acoustic 3.0 model on the LiteRT interpreter.

The official 3.0 TFLite export leaves its input length open and stores one
sample in its place. LiteRT 2.2.0 and later refuse to allocate that shape, so
allocating before the first resize failed every run on them. These tests run
the real file on whatever LiteRT is installed; the order of resize and
allocation is pinned without one in
``core_py/backends_py/test_tf_backend_allocation.py`` and
``acoustic_models/custom_tflite_py/test_inspect_custom_tflite_file.py``.
"""

import numpy as np
import pytest

import birdnet
from birdnet_tests.helper import ensure_litert_or_skip

pytestmark = pytest.mark.litert

_SAMPLE_RATE = 32_000


def _two_segments() -> tuple[np.ndarray, int]:
  rng = np.random.default_rng(11)
  samples = rng.standard_normal(6 * _SAMPLE_RATE).astype(np.float32) * 0.1
  return samples, _SAMPLE_RATE


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_acoustic_v3_0_predicts_and_encodes_on_litert(precision: str) -> None:
  ensure_litert_or_skip()
  model = birdnet.load("acoustic", "3.0", "tf", precision=precision, library="litert")

  predictions = model.predict_arrays(_two_segments(), n_workers=1)
  embeddings = model.encode_arrays(_two_segments(), n_workers=1)

  assert predictions.species_probs.shape == (1, 2, 5)
  assert embeddings.embeddings.shape == (1, 2, 1280)
  assert np.isfinite(np.asarray(embeddings.embeddings)).all()


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_the_official_file_passes_the_custom_classifier_inspection_on_litert(
  precision: str,
) -> None:
  ensure_litert_or_skip()
  model = birdnet.load("acoustic", "3.0", "tf", precision=precision, library="litert")

  info = birdnet.inspect_custom_tflite(model.model_path, library="litert")

  assert info == birdnet.CustomTFLiteInfo("3.0", "replace", 11_725, True, True, None)
