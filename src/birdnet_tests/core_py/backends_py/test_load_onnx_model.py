from pathlib import Path

import pytest

from birdnet.core.backends import load_onnx_model
from birdnet_tests.helper import ensure_onnxruntime_or_skip

pytestmark = pytest.mark.no_tf


def _example_model() -> Path:
  from onnxruntime.datasets import get_example

  return Path(get_example("sigmoid.onnx"))


def test_cpu_session_uses_the_given_thread_count() -> None:
  ensure_onnxruntime_or_skip()

  session = load_onnx_model(_example_model(), "CPU", cpu_threads=3)

  assert session.get_session_options().intra_op_num_threads == 3


def test_without_a_thread_count_the_runtime_default_stays() -> None:
  ensure_onnxruntime_or_skip()

  session = load_onnx_model(_example_model(), "CPU")

  assert session.get_session_options().intra_op_num_threads == 0
