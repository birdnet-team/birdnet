import multiprocessing
import multiprocessing.synchronize

import pytest

from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import (
  assert_encoding_result_is_close,
  assert_encoding_result_is_equal,
  ensure_gpu_or_skip,
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import (
  TEST_FILE_SHORT,
)


def test_pb_cpu_fp32() -> None:
  model = load_perch_v2("CPU")
  with model.encode_session(n_workers=1, device="CPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == (1, 2, 1536)


def test_pb_cpu_fp32_speed_factor() -> None:
  model = load_perch_v2("CPU")
  with model.encode_session(n_workers=1, device="CPU", speed=0.5) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == (1, 3, 1536)


@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == (1, 2, 1536)


def run_session_process(
  x: multiprocessing.synchronize.Barrier, q: multiprocessing.Queue
) -> None:
  model = load_perch_v2("CPU")
  x.wait()
  with model.encode_session(n_workers=1) as session:
    result = session.run(TEST_FILE_SHORT)
  q.put(result)


def test_twice_two_sessions_parallel_processes_fork() -> None:
  use_fork_or_skip()

  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    q = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(x, q))
    p2 = multiprocessing.Process(target=run_session_process, args=(x, q))

    p1.start()
    p2.start()

    res1 = q.get(timeout=None)
    res2 = q.get(timeout=None)

    p1.join()
    p2.join()

  assert_encoding_result_is_equal(res1, res2)


def test_twice_two_sessions_parallel_processes_forkserver() -> None:
  use_forkserver_or_skip()

  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    q = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(x, q))
    p2 = multiprocessing.Process(target=run_session_process, args=(x, q))

    p1.start()
    p2.start()

    res1 = q.get(timeout=None)
    res2 = q.get(timeout=None)

    p1.join()
    p2.join()

  assert_encoding_result_is_equal(res1, res2)


def test_twice_two_sessions_parallel_processes_spawn() -> None:
  use_spawn_or_skip()

  with multiprocessing.Manager() as manager:
    barrier = manager.Barrier(2)
    q = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(barrier, q))
    p2 = multiprocessing.Process(target=run_session_process, args=(barrier, q))

    p1.start()
    p2.start()

    res1 = q.get(timeout=None)
    res2 = q.get(timeout=None)

    p1.join()
    p2.join()

  assert_encoding_result_is_equal(res1, res2)


def test_twice_same_session() -> None:
  model = load_perch_v2("CPU")
  with model.encode_session(n_workers=1) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_encoding_result_is_equal(res1, res2)


def test_twice_two_sessions() -> None:
  model = load_perch_v2("CPU")
  with model.encode_session(n_workers=1) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.encode_session(n_workers=1) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_encoding_result_is_equal(res1, res2)


@pytest.mark.gpu
def test_twice_two_sessions_gpu() -> None:
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.encode_session(n_workers=1, device="GPU") as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_encoding_result_is_close(res1, res2, max_abs_diff=1e-6)


@pytest.mark.gpu
def test_twice_same_session_gpu() -> None:
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_encoding_result_is_close(res1, res2, max_abs_diff=1e-6)
