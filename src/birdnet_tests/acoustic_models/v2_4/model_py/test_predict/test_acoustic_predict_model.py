import multiprocessing
import multiprocessing.synchronize
import threading
from queue import Queue

import numpy

from birdnet.model_loader import load
from birdnet_tests.helper import (
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_WAV


def test_pb_fp32() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  res = model.predict(TEST_FILE_WAV, n_workers=1)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=1)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=1)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06305, decimal=4)


def test_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06216, decimal=4)


def test_litert_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_litert_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06305, decimal=4)


def test_litert_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06354, decimal=4)


def test_tf_fp32_twice_two_sessions() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def run_session(
  x: multiprocessing.synchronize.Barrier, queue: multiprocessing.Queue
) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  x.wait()
  with model.predict_session(n_workers=1) as session:
    result = session.run(TEST_FILE_WAV)
    queue.put(result)


def test_tf_fp32_twice_two_sessions_parallel_processes_fork() -> None:
  use_fork_or_skip()
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session, args=(x, queue))

    p1.start()
    p2.start()

    res = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)

  mean = res2.species_probs.mean()
  assert res2.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_tf_fp32_twice_two_sessions_parallel_processes_forkserver() -> None:
  use_forkserver_or_skip()
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session, args=(x, queue))

    p1.start()
    p2.start()

    res = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)

  mean = res2.species_probs.mean()
  assert res2.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_tf_fp32_twice_two_sessions_parallel_processes_spawn() -> None:
  use_spawn_or_skip()
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session, args=(x, queue))

    p1.start()
    p2.start()

    res = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)

  mean = res2.species_probs.mean()
  assert res2.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def run_session_thread(barrier: threading.Barrier, queue: Queue) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  barrier.wait()
  with model.predict_session(n_workers=1) as session:
    result = session.run(TEST_FILE_WAV)
    queue.put(result)


def test_tf_fp32_twice_two_sessions_parallel_threads() -> None:
  barrier = threading.Barrier(2)
  queue = Queue()

  t1 = threading.Thread(target=run_session_thread, args=(barrier, queue))
  t2 = threading.Thread(target=run_session_thread, args=(barrier, queue))

  t1.start()
  t2.start()

  res = queue.get(timeout=None)
  res2 = queue.get(timeout=None)

  t1.join()
  t2.join()

  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)

  mean = res2.species_probs.mean()
  assert res2.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_tf_fp32_twice_same_session() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_pb_fp32_twice_two_sessions() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_pb_fp32_twice_same_session() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1) as session:
    res = session.run(TEST_FILE_WAV)
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)
