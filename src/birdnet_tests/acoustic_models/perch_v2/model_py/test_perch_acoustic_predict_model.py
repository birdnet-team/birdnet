"""Perch's own prediction shape on the real model.

Everything the pipeline does with it (callbacks, repeated runs, the softmax)
is pinned on the fake backend under ``acoustic_models/inference/``.
"""

import numpy
import pytest

from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  ensure_gpu_or_skip,
  ensure_not_intel_macos_or_skip,
  ensure_tf_2_20_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT


@pytest.fixture(autouse=True)
def _ensure_supported_tf() -> None:
  ensure_tf_2_20_or_skip()


def test_cpu() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(n_workers=1, top_k=None, device="CPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 2, 14795)


@pytest.mark.gpu
def test_gpu() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.predict_session(n_workers=1, top_k=None, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 2, 14795)


@pytest.mark.gpu
@pytest.mark.usefixtures("deterministic_tf_ops")
def test_twice_two_sessions_gpu() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.predict_session(
    n_workers=1, device="GPU", top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.predict_session(
    n_workers=1, device="GPU", top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_close(res1, res2, max_abs_diff=1e-5)


@pytest.mark.gpu
def test_twice_same_session_gpu() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.predict_session(
    n_workers=1, device="GPU", top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_close(res1, res2, max_abs_diff=1e-5)
