import math

import pytest

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet_tests.acoustic_models.v3_0.model_py.conftest import (
  V3_0Backend,
  load_v3_0,
  onnx_fp32_prediction,
)
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  ensure_onnxruntime_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT, TEST_FILE_SHORT_DURATION_S

_N_CLASSES = AcousticModelV3_0.get_n_classes()
_DEFAULT_SCORE_SHAPE = (1, 3, _N_CLASSES)
_PT_ONNX_PREDICTION_MAX_ABS_DIFF = 1e-4
_PT_ONNX_PREDICTION_MEAN_ABS_DIFF = 1e-5
_PB_ONNX_PREDICTION_MAX_ABS_DIFF = 1e-4
_PB_ONNX_PREDICTION_MEAN_ABS_DIFF = 1e-5


def _predict(
  model: AcousticModelV3_0, segment_size_s: float = 3.0
) -> AcousticPredictionResultBase:
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-float("inf"),
    segment_size_s=segment_size_s,
  ) as session:
    return session.run(TEST_FILE_SHORT)


def _onnx_prediction(segment_size_s: float) -> AcousticPredictionResultBase:
  if segment_size_s == 3.0:
    return onnx_fp32_prediction()
  return _predict(load_v3_0("onnx"), segment_size_s)


@pytest.mark.no_tf
@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_v3_0_predict_default_file_shape(backend: V3_0Backend) -> None:
  res = _onnx_prediction(3.0) if backend == "onnx" else _predict(load_v3_0(backend))

  assert res.species_probs.shape == _DEFAULT_SCORE_SHAPE
  assert res.segment_duration_s == 3.0


# A segment length that does not divide the file is the producer's arithmetic
# (core/producer_py/test_get_segments_with_overlap_samples.py); the model sees
# the same dynamic-width path either way.
@pytest.mark.no_tf
@pytest.mark.parametrize("backend", ["pt", "onnx"])
@pytest.mark.parametrize(("segment_size_s", "expected_segments"), [(2.0, 4), (4.0, 2)])
def test_v3_0_predict_respects_segment_size(
  backend: V3_0Backend,
  segment_size_s: float,
  expected_segments: int,
) -> None:
  model = load_v3_0(backend)
  res = model.predict(
    TEST_FILE_SHORT,
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-float("inf"),
    segment_size_s=segment_size_s,
  )

  assert res.species_probs.shape == (1, expected_segments, _N_CLASSES)
  assert res.segment_duration_s == segment_size_s


def test_v3_0_predict_pb_and_onnx_are_close() -> None:
  # Runs the real SavedModel end to end: stale signature names or keys, which
  # no load-only test can catch, fail here against onnx.
  ensure_onnxruntime_or_skip()
  pb_result = _predict(load_v3_0("pb"))
  onnx_result = onnx_fp32_prediction()

  assert pb_result.species_probs.shape == _DEFAULT_SCORE_SHAPE
  assert_prediction_result_is_close(
    pb_result,
    onnx_result,
    max_abs_diff=_PB_ONNX_PREDICTION_MAX_ABS_DIFF,
    mean_abs_diff=_PB_ONNX_PREDICTION_MEAN_ABS_DIFF,
  )


@pytest.mark.no_tf
@pytest.mark.parametrize("segment_size_s", [2.0, 3.0, 4.0, 5.0])
def test_v3_0_predict_pt_and_onnx_are_close_with_custom_segment_size(
  segment_size_s: float,
) -> None:
  ensure_onnxruntime_or_skip()
  pt_result = _predict(load_v3_0("pt"), segment_size_s)
  onnx_result = _onnx_prediction(segment_size_s)

  segments_cnt = math.ceil(TEST_FILE_SHORT_DURATION_S / segment_size_s)
  assert pt_result.species_probs.shape == (1, segments_cnt, _N_CLASSES)
  assert onnx_result.species_probs.shape == (1, segments_cnt, _N_CLASSES)
  assert_prediction_result_is_close(
    pt_result,
    onnx_result,
    max_abs_diff=_PT_ONNX_PREDICTION_MAX_ABS_DIFF,
    mean_abs_diff=_PT_ONNX_PREDICTION_MEAN_ABS_DIFF,
  )
