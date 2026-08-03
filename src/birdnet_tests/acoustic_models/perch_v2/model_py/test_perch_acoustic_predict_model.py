import multiprocessing
import multiprocessing.synchronize

import numpy
import pytest

from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  assert_prediction_result_is_equal,
  ensure_gpu_or_skip,
  ensure_not_intel_macos_or_skip,
  ensure_not_mac_or_skip,
  ensure_tf_2_20_or_skip,
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT


@pytest.fixture(autouse=True)
def _ensure_supported_tf() -> None:
  ensure_tf_2_20_or_skip()


def test_cpu() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(n_workers=1, top_k=None, device="CPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 2, 14795)


def test_cpu_speed_factor() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(
    n_workers=1, top_k=None, device="CPU", speed=0.5
  ) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 3, 14795)


def test_cpu_softmax_probs_sum_to_one_per_segment() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(
    n_workers=1,
    top_k=None,
    device="CPU",
    default_confidence_threshold=-numpy.inf,
    apply_softmax=True,
  ) as session:
    res = session.run(TEST_FILE_SHORT)

  assert res.species_probs.shape == (1, 2, 14795)
  assert numpy.all(res.species_probs > 0)
  assert numpy.all(res.species_probs <= 1)
  numpy.testing.assert_allclose(
    res.species_probs.sum(axis=-1), numpy.ones((1, 2)), atol=1e-4
  )


def test_cpu_softmax_keeps_ranking_of_logits() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(
    n_workers=1, top_k=1, device="CPU", default_confidence_threshold=-numpy.inf
  ) as session:
    res_logits = session.run(TEST_FILE_SHORT)
  with model.predict_session(
    n_workers=1,
    top_k=1,
    device="CPU",
    default_confidence_threshold=-numpy.inf,
    apply_softmax=True,
  ) as session:
    res_softmax = session.run(TEST_FILE_SHORT)

  # softmax is monotonic, so the top species must not change
  numpy.testing.assert_array_equal(res_softmax.species_ids, res_logits.species_ids)


@pytest.mark.gpu
def test_gpu() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  with model.predict_session(n_workers=1, top_k=None, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 2, 14795)


@pytest.mark.gpu
def xtest_gpu_too_large_batch_size_raises_error() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_gpu_or_skip()

  import soundfile as sf

  sf_data, sr = sf.read(TEST_FILE_LONG, dtype="float32")
  data_6h = numpy.tile(sf_data, 30 * 6)

  model = load_perch_v2("GPU")

  with (
    pytest.raises(RuntimeError),
    model.predict_session(
      n_workers=1, top_k=None, device="GPU", batch_size=2000
    ) as session,
  ):
    session.run_arrays((data_6h, sr))


def run_session_process(
  x: multiprocessing.synchronize.Barrier, q: multiprocessing.Queue
) -> None:
  model = load_perch_v2("CPU")
  x.wait()
  with model.predict_session(n_workers=1, top_k=None) as session:
    result = session.run(TEST_FILE_SHORT)
  q.put(result)


@pytest.mark.fork
def test_twice_two_sessions_parallel_processes_fork() -> None:
  ensure_not_mac_or_skip()  # reason unknown why this hangs on macOS
  ensure_not_intel_macos_or_skip()
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

  assert_prediction_result_is_equal(res1, res2)


def test_twice_two_sessions_parallel_processes_forkserver() -> None:
  ensure_not_intel_macos_or_skip()
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

  assert_prediction_result_is_equal(res1, res2)


def test_twice_two_sessions_parallel_processes_spawn() -> None:
  ensure_not_intel_macos_or_skip()
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

  assert_prediction_result_is_equal(res1, res2)


def test_twice_same_session() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


def test_twice_two_sessions() -> None:
  ensure_not_intel_macos_or_skip()

  model = load_perch_v2("CPU")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None) as session:
    res2 = session.run(TEST_FILE_SHORT)
  assert_prediction_result_is_equal(res1, res2)


@pytest.mark.gpu
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
