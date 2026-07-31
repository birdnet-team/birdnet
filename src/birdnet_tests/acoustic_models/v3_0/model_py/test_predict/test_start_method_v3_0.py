import multiprocessing
import multiprocessing.synchronize
from typing import Literal

import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import (
  assert_prediction_result_is_equal,
  ensure_not_intel_macos_or_skip,
  ensure_not_mac_or_skip,
  ensure_onnxruntime_or_skip,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

_Backend = Literal["pt", "onnx"]


def _skip_unless_backend_available(backend: _Backend) -> None:
  if backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
  else:
    ensure_onnxruntime_or_skip()


def _run_session_process(
  backend: _Backend,
  x: multiprocessing.synchronize.Barrier,
  q: multiprocessing.Queue,
) -> None:
  model = load("acoustic", "3.0", backend, precision="fp32")
  x.wait()
  with model.predict_session(
    n_workers=1, top_k=None, default_confidence_threshold=-float("inf")
  ) as session:
    result = session.run(TEST_FILE_SHORT)
  q.put(result)


def _run_two_parallel_sessions(backend: _Backend) -> None:
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    q = manager.Queue()

    p1 = multiprocessing.Process(target=_run_session_process, args=(backend, x, q))
    p2 = multiprocessing.Process(target=_run_session_process, args=(backend, x, q))

    p1.start()
    p2.start()

    res1 = q.get(timeout=None)
    res2 = q.get(timeout=None)

    p1.join()
    p2.join()

  assert_prediction_result_is_equal(res1, res2)


# The v3.0 "pt" (PyTorch) and "onnx" backends carry no TensorFlow runtime. Unlike
# the TF/TFLite/PB backends, whose fork lanes wedge on the fork-after-TensorFlow
# deadlock, they can be forked safely. This is the start method birdnet leans on as
# it moves from TensorFlow to PyTorch, so these lanes must stay green; they are the
# coverage that proves fork works for everything except TensorFlow.
@pytest.mark.fork
@pytest.mark.fork_nontf
@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_twice_two_sessions_parallel_processes_fork(backend: _Backend) -> None:
  # macOS is skipped to match the perch/v2.4 fork tests: forking the
  # multiprocessing.Manager helper processes hangs there for reasons unrelated to
  # the backend. Linux is where birdnet forks in production and where CI exercises
  # this lane, so the non-TF fork guarantee is covered there.
  ensure_not_mac_or_skip()
  ensure_not_intel_macos_or_skip()
  _skip_unless_backend_available(backend)
  use_fork_or_skip()
  _run_two_parallel_sessions(backend)


@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_twice_two_sessions_parallel_processes_forkserver(backend: _Backend) -> None:
  ensure_not_intel_macos_or_skip()
  _skip_unless_backend_available(backend)
  use_forkserver_or_skip()
  _run_two_parallel_sessions(backend)


@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_twice_two_sessions_parallel_processes_spawn(backend: _Backend) -> None:
  ensure_not_intel_macos_or_skip()
  _skip_unless_backend_available(backend)
  use_spawn_or_skip()
  _run_two_parallel_sessions(backend)
