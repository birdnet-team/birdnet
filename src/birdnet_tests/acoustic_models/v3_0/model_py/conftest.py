"""The loader and the shared onnx runs of the V3.0 model tests.

Importable as ``birdnet_tests.acoustic_models.v3_0.model_py.conftest``. The onnx
fp32 default runs are cached: one per pytest process, however many tests read
from them (under xdist each worker holds its own, so the saving is partial).
"""

from __future__ import annotations

import functools
from typing import Literal

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_onnxruntime_or_skip,
  ensure_tf_2_18_or_skip,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

V3_0Backend = Literal["pt", "onnx", "tf", "pb"]


def load_v3_0(backend: V3_0Backend, precision: str = "fp32") -> AcousticModelV3_0:
  if backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
    return load("acoustic", "3.0", "pt", precision=precision)
  if backend == "onnx":
    ensure_onnxruntime_or_skip()
    return load("acoustic", "3.0", "onnx", precision=precision)
  # The v3.0 exports need a newer TF than the macOS Intel pin (<2.17).
  ensure_tf_2_18_or_skip()
  if backend == "tf":
    return load("acoustic", "3.0", "tf", precision=precision, library="tflite")
  return load("acoustic", "3.0", "pb", precision=precision)


@functools.cache
def onnx_fp32_prediction() -> AcousticPredictionResultBase:
  """The onnx fp32 model on TEST_FILE_SHORT, every score of every segment."""
  with load_v3_0("onnx").predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-float("inf"),
  ) as session:
    return session.run(TEST_FILE_SHORT)


@functools.cache
def onnx_fp32_encoding() -> AcousticEncodingResultBase:
  """The onnx fp32 model's embeddings of TEST_FILE_SHORT."""
  with load_v3_0("onnx").encode_session(n_workers=1) as session:
    return session.run(TEST_FILE_SHORT)
