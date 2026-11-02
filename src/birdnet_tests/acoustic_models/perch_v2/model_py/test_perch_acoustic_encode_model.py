"""Perch's own embedding shape on the real model.

Everything the pipeline does with it (callbacks, repeated runs) is pinned on
the fake backend under ``acoustic_models/inference/``.
"""

import pytest

from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import (
  assert_encoding_result_is_close,
  ensure_gpu_or_skip,
  ensure_not_intel_macos_or_skip,
  ensure_tf_2_20_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT


@pytest.fixture(autouse=True)
def _ensure_supported_tf() -> None:
  ensure_tf_2_20_or_skip()


def test_pb_cpu_fp32() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.encode_session(n_workers=1, device="CPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == (1, 2, 1536)


@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == (1, 2, 1536)


@pytest.mark.gpu
def test_twice_two_sessions_gpu() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.encode_session(n_workers=1, device="GPU") as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_encoding_result_is_close(res1, res2, max_abs_diff=1e-6)


@pytest.mark.gpu
def test_twice_same_session_gpu() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_encoding_result_is_close(res1, res2, max_abs_diff=1e-6)
