# ruff: noqa: F841
import multiprocessing
import multiprocessing.synchronize

import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_litert_or_skip,
  join_children_or_fail,
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)


def test_pb_cpu_fp32() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None, device="CPU") as session:
    pass


@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None, device="GPU") as session:
    pass


def test_tflite_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


def test_tflite_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


def test_tflite_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


@pytest.mark.litert
def test_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


@pytest.mark.litert
def test_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


def test_tflite_fp32_twice_two_sessions() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session1:
    pass
  with model.predict_session(n_workers=1, top_k=None) as session2:
    pass


@pytest.mark.litert
def test_litert_fp32_twice_two_sessions() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session1:
    pass
  with model.predict_session(n_workers=1, top_k=None) as session2:
    pass


def run_session_process(barrier: multiprocessing.synchronize.Barrier) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  barrier.wait()
  with model.predict_session(n_workers=1, top_k=None) as session:
    pass


@pytest.mark.fork
def test_tflite_fp32_twice_two_sessions_parallel_processes_fork() -> None:
  use_fork_or_skip()

  with multiprocessing.Manager() as manager:
    barrier = manager.Barrier(2)

    p1 = multiprocessing.Process(target=run_session_process, args=(barrier,))
    p2 = multiprocessing.Process(target=run_session_process, args=(barrier,))

    p1.start()
    p2.start()

    join_children_or_fail(p1, p2)


def test_tflite_fp32_twice_two_sessions_parallel_processes_forkserver() -> None:
  use_forkserver_or_skip()

  with multiprocessing.Manager() as manager:
    barrier = manager.Barrier(2)

    p1 = multiprocessing.Process(target=run_session_process, args=(barrier,))
    p2 = multiprocessing.Process(target=run_session_process, args=(barrier,))

    p1.start()
    p2.start()

    join_children_or_fail(p1, p2)


def test_tflite_fp32_twice_two_sessions_parallel_processes_spawn() -> None:
  use_spawn_or_skip()

  with multiprocessing.Manager() as manager:
    barrier = manager.Barrier(2)

    p1 = multiprocessing.Process(target=run_session_process, args=(barrier,))
    p2 = multiprocessing.Process(target=run_session_process, args=(barrier,))

    p1.start()
    p2.start()

    join_children_or_fail(p1, p2)


def test_pb_cpu_fp32_two_sessions() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None) as session1:
    pass
  with model.predict_session(n_workers=1, top_k=None) as session2:
    pass


@pytest.mark.gpu
def test_pb_gpu_fp32_two_sessions() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, device="GPU") as session1:
    pass
  with model.predict_session(n_workers=1, device="GPU") as session2:
    pass
