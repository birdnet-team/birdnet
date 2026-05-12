from typing import Literal

import pytest
import soundfile

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import (
  assert_encoding_result_is_close,
  ensure_onnxruntime_or_skip,
  ensure_torch_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

_Backend = Literal["pt", "onnx"]
_DEFAULT_EMB_SHAPE = (1, 3, 1280)
_TWO_ARRAY_EMB_SHAPE = (2, 3, 1280)
_PT_ONNX_ENCODING_MAX_ABS_DIFF = 0.5
_PT_ONNX_ENCODING_MEAN_ABS_DIFF = 0.02


def _load_model(backend: _Backend) -> AcousticModelV3_0:
  if backend == "pt":
    ensure_torch_or_skip()
  else:
    ensure_onnxruntime_or_skip()

  return load("acoustic", "3.0", backend, precision="fp32")


@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_v3_0_encode_default_file_shape(backend: _Backend) -> None:
  model = _load_model(backend)
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)

  assert res.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert res.segment_duration_s == 3.0


@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_v3_0_encode_default_np_array_shape(backend: _Backend) -> None:
  audio = soundfile.read(TEST_FILE_SHORT)
  model = _load_model(backend)
  with model.encode_session(n_workers=1) as session:
    res = session.run_arrays(audio)

  assert res.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert res.segment_duration_s == 3.0


@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_v3_0_encode_two_np_arrays_shape(backend: _Backend) -> None:
  audio = soundfile.read(TEST_FILE_SHORT)
  model = _load_model(backend)
  with model.encode_session(n_workers=1) as session:
    res = session.run_arrays([audio, audio])

  assert res.embeddings.shape == _TWO_ARRAY_EMB_SHAPE
  assert res.segment_duration_s == 3.0


@pytest.mark.parametrize("backend", ["pt", "onnx"])
@pytest.mark.parametrize(
  ("segment_size_s", "expected_segments"),
  [(2.0, 4), (4.0, 2)],
)
def test_v3_0_encode_respects_segment_size(
  backend: _Backend,
  segment_size_s: float,
  expected_segments: int,
) -> None:
  model = _load_model(backend)
  res = model.encode(
    TEST_FILE_SHORT,
    n_workers=1,
    segment_size_s=segment_size_s,
  )

  assert res.embeddings.shape == (1, expected_segments, 1280)
  assert res.segment_duration_s == segment_size_s


def test_v3_0_encode_pt_and_onnx_are_close() -> None:
  ensure_torch_or_skip()
  ensure_onnxruntime_or_skip()

  pt_model = load("acoustic", "3.0", "pt", precision="fp32")
  onnx_model = load("acoustic", "3.0", "onnx", precision="fp32")

  with pt_model.encode_session(n_workers=1) as pt_session:
    pt_result = pt_session.run(TEST_FILE_SHORT)

  with onnx_model.encode_session(n_workers=1) as onnx_session:
    onnx_result = onnx_session.run(TEST_FILE_SHORT)

    assert_encoding_result_is_close(
      pt_result,
      onnx_result,
      max_abs_diff=_PT_ONNX_ENCODING_MAX_ABS_DIFF,
      mean_abs_diff=_PT_ONNX_ENCODING_MEAN_ABS_DIFF,
    )


def test_v3_0_encode_pt_and_onnx_are_close_with_custom_segment_size() -> None:
  ensure_torch_or_skip()
  ensure_onnxruntime_or_skip()

  pt_model = load("acoustic", "3.0", "pt", precision="fp32")
  onnx_model = load("acoustic", "3.0", "onnx", precision="fp32")

  with pt_model.encode_session(n_workers=1, segment_size_s=2.0) as pt_session:
    pt_result = pt_session.run(TEST_FILE_SHORT)

  with onnx_model.encode_session(n_workers=1, segment_size_s=2.0) as onnx_session:
    onnx_result = onnx_session.run(TEST_FILE_SHORT)

  assert pt_result.embeddings.shape == (1, 4, 1280)
  assert onnx_result.embeddings.shape == (1, 4, 1280)
  assert_encoding_result_is_close(
    pt_result,
    onnx_result,
    max_abs_diff=_PT_ONNX_ENCODING_MAX_ABS_DIFF,
    mean_abs_diff=_PT_ONNX_ENCODING_MEAN_ABS_DIFF,
  )
