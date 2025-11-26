import multiprocessing
import multiprocessing.synchronize
import queue
import threading

import numpy
import pytest
import soundfile as sf

from birdnet.model_loader import load
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  assert_prediction_result_is_equal,
  ensure_gpu_or_skip,
  ensure_litert_or_skip,
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import (
  TEST_FILE_LONG,
  TEST_FILE_SHORT,
  TEST_FILE_SHORT_SCORE_SHAPE,
)


def test_pb_cpu_fp32() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None, device="CPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_pb_cpu_fp32_speed_factor() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(
    n_workers=1, top_k=None, device="CPU", speed=0.5
  ) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 5, 6522)


@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tf_fp32_np_array() -> None:
  sf_read = sf.read(TEST_FILE_SHORT)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run_arrays(sf_read)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tf_fp32_two_np_arrays() -> None:
  sf_read = sf.read(TEST_FILE_SHORT)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run_arrays([sf_read, sf_read])
  assert res.species_probs.shape == (2, 3, 6522)


def xtest_tf_fp32_large_np_array() -> None:
  sf_data, sr = sf.read(TEST_FILE_LONG, dtype="float32")
  data_6h = numpy.tile(sf_data, 30 * 6)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(top_k=None) as session:
    res = session.run_arrays((data_6h, sr))
  assert res.species_probs.shape == (1, 40 * 30 * 6, 6522)


def test_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tf_int8_all_species_no_threshold_should_not_mask_anything() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-numpy.inf,
    apply_sigmoid=False,
  ) as session:
    res = session.run(TEST_FILE_SHORT)
  assert numpy.all(~res.species_masked)


@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


@pytest.mark.litert
def test_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


@pytest.mark.litert
def test_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tf_fp32_twice_two_sessions() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


@pytest.mark.litert
def test_litert_fp32_twice_two_sessions() -> None:
  ensure_litert_or_skip()
  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


def run_session(
  x: multiprocessing.synchronize.Barrier, queue: multiprocessing.Queue
) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  x.wait()
  with model.predict_session(n_workers=1, top_k=None) as session:
    result = session.run(TEST_FILE_SHORT)
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

    res1 = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  assert_prediction_result_is_equal(res1, res2)


def test_tf_fp32_twice_two_sessions_parallel_processes_forkserver() -> None:
  use_forkserver_or_skip()
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session, args=(x, queue))

    p1.start()
    p2.start()

    res1 = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  assert_prediction_result_is_equal(res1, res2)


def test_tf_fp32_twice_two_sessions_parallel_processes_spawn() -> None:
  use_spawn_or_skip()
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session, args=(x, queue))

    p1.start()
    p2.start()

    res1 = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  assert_prediction_result_is_equal(res1, res2)


def run_session_thread(barrier: threading.Barrier, queue: queue.Queue) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  barrier.wait()
  with model.predict_session(n_workers=1, top_k=None) as session:
    result = session.run(TEST_FILE_SHORT)
    queue.put(result)


def test_tf_fp32_twice_two_sessions_parallel_threads() -> None:
  barrier = threading.Barrier(2)
  m = multiprocessing.Manager()
  queue = m.Queue()

  t1 = threading.Thread(target=run_session_thread, args=(barrier, queue))
  t2 = threading.Thread(target=run_session_thread, args=(barrier, queue))

  t1.start()
  t2.start()

  res1 = queue.get(timeout=None)
  res2 = queue.get(timeout=None)

  t1.join()
  t2.join()

  assert_prediction_result_is_equal(res1, res2)


def test_tf_fp32_twice_same_session() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


def test_pb_cpu_fp32_twice_two_sessions() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


@pytest.mark.gpu
def test_pb_gpu_fp32_twice_two_sessions() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(
    n_workers=1, device="GPU", top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.predict_session(
    n_workers=1, device="GPU", top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_close(res1, res2, max_abs_diff=1e-6)


def test_pb_cpu_fp32_twice_same_session() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


@pytest.mark.gpu
def test_pb_gpu_fp32_twice_same_session() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(
    n_workers=1, device="GPU", top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_close(res1, res2, max_abs_diff=1e-6)
