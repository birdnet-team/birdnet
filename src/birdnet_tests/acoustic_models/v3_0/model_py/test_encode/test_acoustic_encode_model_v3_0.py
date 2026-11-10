import math

import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet_tests.acoustic_models.v3_0.model_py.conftest import (
  V3_0Backend,
  load_v3_0,
  onnx_fp32_encoding,
)
from birdnet_tests.helper import (
  assert_encoding_result_is_close,
  ensure_onnxruntime_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT, TEST_FILE_SHORT_DURATION_S

_DEFAULT_EMB_SHAPE = (1, 3, 1280)
_PT_ONNX_ENCODING_MAX_ABS_DIFF = 1e-4
_PT_ONNX_ENCODING_MEAN_ABS_DIFF = 1e-5
_PB_ONNX_ENCODING_MAX_ABS_DIFF = 1e-4
_PB_ONNX_ENCODING_MEAN_ABS_DIFF = 1e-5
_TF_ONNX_ENCODING_MAX_ABS_DIFF = 1e-4
_TF_ONNX_ENCODING_MEAN_ABS_DIFF = 1e-5
# On the silence-padded final segment the per-segment normalization amplifies
# fp16 rounding to ~0.033 (both fp16 exports; <=0.005 on real audio). The wide
# max absorbs that; the tight mean still rejects a wrong tensor.
_FP16_ENCODING_MAX_ABS_DIFF = 5e-2
_FP16_ENCODING_MEAN_ABS_DIFF = 2e-3


def _encode(
  model: AcousticModelV3_0, segment_size_s: float = 3.0
) -> AcousticEncodingResultBase:
  with model.encode_session(n_workers=1, segment_size_s=segment_size_s) as session:
    return session.run(TEST_FILE_SHORT)


def _onnx_encoding(segment_size_s: float) -> AcousticEncodingResultBase:
  if segment_size_s == 3.0:
    return onnx_fp32_encoding()
  return _encode(load_v3_0("onnx"), segment_size_s)


@pytest.mark.no_tf
@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_v3_0_encode_default_file_shape(backend: V3_0Backend) -> None:
  res = _onnx_encoding(3.0) if backend == "onnx" else _encode(load_v3_0(backend))

  assert res.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert res.segment_duration_s == 3.0


# A segment length that does not divide the file is the producer's arithmetic
# (core/producer_py/test_get_segments_with_overlap_samples.py); the model sees
# the same dynamic-width path either way.
@pytest.mark.no_tf
@pytest.mark.parametrize("backend", ["pt", "onnx"])
@pytest.mark.parametrize(("segment_size_s", "expected_segments"), [(2.0, 4), (4.0, 2)])
def test_v3_0_encode_respects_segment_size(
  backend: V3_0Backend,
  segment_size_s: float,
  expected_segments: int,
) -> None:
  model = load_v3_0(backend)
  res = model.encode(
    TEST_FILE_SHORT,
    n_workers=1,
    segment_size_s=segment_size_s,
  )

  assert res.embeddings.shape == (1, expected_segments, 1280)
  assert res.segment_duration_s == segment_size_s


def test_v3_0_encode_pb_and_onnx_are_close() -> None:
  # The v3.0 SavedModel has no separate "embeddings" signature; embeddings
  # come from "serving_default".
  ensure_onnxruntime_or_skip()
  pb_result = _encode(load_v3_0("pb"))
  onnx_result = onnx_fp32_encoding()

  assert pb_result.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert_encoding_result_is_close(
    pb_result,
    onnx_result,
    max_abs_diff=_PB_ONNX_ENCODING_MAX_ABS_DIFF,
    mean_abs_diff=_PB_ONNX_ENCODING_MEAN_ABS_DIFF,
  )


def test_v3_0_encode_tf_and_onnx_are_close() -> None:
  # The TFLite backend reads embeddings by tensor index; a wrong one returns
  # another tensor silently.
  ensure_onnxruntime_or_skip()
  tf_result = _encode(load_v3_0("tf"))
  onnx_result = onnx_fp32_encoding()

  assert tf_result.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert_encoding_result_is_close(
    tf_result,
    onnx_result,
    max_abs_diff=_TF_ONNX_ENCODING_MAX_ABS_DIFF,
    mean_abs_diff=_TF_ONNX_ENCODING_MEAN_ABS_DIFF,
  )


def test_v3_0_encode_tf_fp16_and_onnx_are_close() -> None:
  # The fp16 TFLite export has its own tensor indices, unexercised elsewhere.
  ensure_onnxruntime_or_skip()
  tf_result = _encode(load_v3_0("tf", "fp16"))
  onnx_result = onnx_fp32_encoding()

  assert tf_result.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert_encoding_result_is_close(
    tf_result,
    onnx_result,
    max_abs_diff=_FP16_ENCODING_MAX_ABS_DIFF,
    mean_abs_diff=_FP16_ENCODING_MEAN_ABS_DIFF,
  )


@pytest.mark.no_tf
def test_v3_0_encode_onnx_fp16_and_fp32_are_close() -> None:
  # The fp16 onnx export has its own encoding_out_idx, unexercised elsewhere.
  fp16_result = _encode(load_v3_0("onnx", "fp16"))
  fp32_result = onnx_fp32_encoding()

  assert fp16_result.embeddings.shape == _DEFAULT_EMB_SHAPE
  assert_encoding_result_is_close(
    fp16_result,
    fp32_result,
    max_abs_diff=_FP16_ENCODING_MAX_ABS_DIFF,
    mean_abs_diff=_FP16_ENCODING_MEAN_ABS_DIFF,
  )


@pytest.mark.no_tf
@pytest.mark.parametrize("segment_size_s", [2.0, 3.0, 4.0, 5.0])
def test_v3_0_encode_pt_and_onnx_are_close_with_custom_segment_size(
  segment_size_s: float,
) -> None:
  ensure_onnxruntime_or_skip()
  pt_result = _encode(load_v3_0("pt"), segment_size_s)
  onnx_result = _onnx_encoding(segment_size_s)

  segments_cnt = math.ceil(TEST_FILE_SHORT_DURATION_S / segment_size_s)
  assert pt_result.embeddings.shape == (1, segments_cnt, 1280)
  assert onnx_result.embeddings.shape == (1, segments_cnt, 1280)
  assert_encoding_result_is_close(
    pt_result,
    onnx_result,
    max_abs_diff=_PT_ONNX_ENCODING_MAX_ABS_DIFF,
    mean_abs_diff=_PT_ONNX_ENCODING_MEAN_ABS_DIFF,
  )
