import multiprocessing
import multiprocessing.synchronize
import queue
import tempfile
import threading
from pathlib import Path

import numpy
import pytest
import soundfile as sf

from birdnet.acoustic_models.inference.perf_tracker import AcousticProgressStats
from birdnet.model_loader import load
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  assert_prediction_result_is_equal,
  create_zero_len_wav,
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


def xtest_pb_cpu_fp32_callback() -> None:
  # depending on the speed of the machine, this may or may not collect any stats
  collected_stats = []

  def test_callback(data: AcousticProgressStats) -> None:
    assert data is not None
    collected_stats.append(data)

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(
    n_workers=1,
    top_k=None,
    device="CPU",
    progress_callback=test_callback,
    show_stats="progress",
  ) as session:
    res = session.run(TEST_FILE_LONG)
  assert res.species_probs.shape == (1, 40, 6522)
  assert len(collected_stats) > 0


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


def test_tflite_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tflite_fp32_empty_file_is_skipped() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as empty_wav1:
      empty_wav1.write(b"")
    res = session.run(empty_wav1.name)
    empty_wav1.close()
  assert res.species_probs.shape == (1, 0, 6522)
  assert numpy.all(res.species_probs[0] == 0)
  assert numpy.all(res.species_ids[0] == 0)
  assert numpy.all(res.species_masked[0])
  assert res.get_unprocessed_files() == {Path(empty_wav1.name).absolute()}


def test_tflite_fp32_empty_wav_is_not_unprocessed() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as tmp_broken_wav:
      create_zero_len_wav(tmp_broken_wav)
    res = session.run(tmp_broken_wav.name)
    tmp_broken_wav.close()
  assert res.species_probs.shape == (1, 0, 6522)
  assert numpy.all(res.species_probs[0] == 0)
  assert numpy.all(res.species_ids[0] == 0)
  assert numpy.all(res.species_masked[0])
  assert res.get_unprocessed_files() == set()


def test_tflite_fp32_empty_and_valid_file() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1, top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as empty_wav1:
      empty_wav1.write(b"")
    res = session.run([empty_wav1.name, TEST_FILE_SHORT])
    empty_wav1.close()
  assert res.species_probs.shape == (2, 3, 6522)
  unprocessed_idx = res.unprocessable_inputs[0]
  processed_idx = list(set(range(len(res.inputs))) - {unprocessed_idx})[0]
  assert numpy.all(res.species_probs[unprocessed_idx] == 0)
  assert numpy.all(res.species_ids[unprocessed_idx] == 0)
  assert numpy.all(res.species_masked[unprocessed_idx])
  assert numpy.all(res.species_probs[processed_idx] != 0)
  assert not numpy.any(res.species_masked[processed_idx])
  assert res.get_unprocessed_files() == {Path(empty_wav1.name).absolute()}


def test_tflite_fp32_empty_files_are_skipped_normal_is_kept() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1, top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as tmp_broken_wav1:
      tmp_broken_wav1.write(b"NOT_A_VALID_WAV_FILE")
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as tmp_broken_wav2:
      tmp_broken_wav2.write(b"NOT_A_VALID_WAV_FILE")
    res = session.run(
      [
        tmp_broken_wav1.name,
        TEST_FILE_SHORT,
        tmp_broken_wav2.name,
      ]
    )
    tmp_broken_wav1.close()
    tmp_broken_wav2.close()

  assert res.get_unprocessed_files() == {
    Path(tmp_broken_wav1.name).absolute(),
    Path(tmp_broken_wav2.name).absolute(),
  }


def test_tflite_fp32_empty_files_are_skipped_two_feeders() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-numpy.inf,
    n_feeders=2,
  ) as session:
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as empty_wav1:
      empty_wav1.write(b"")
    with tempfile.NamedTemporaryFile(
      suffix=".wav", delete=False, mode="wb"
    ) as empty_wav2:
      empty_wav2.write(b"")
    res = session.run([empty_wav1.name, empty_wav2.name])
    empty_wav1.close()
    empty_wav2.close()

  assert numpy.all(res.species_probs[0] == 0)
  assert numpy.all(res.species_ids[0] == 0)
  assert numpy.all(res.species_masked[0])

  assert numpy.all(res.species_probs[1] == 0)
  assert numpy.all(res.species_ids[1] == 0)
  assert numpy.all(res.species_masked[1])

  assert res.get_unprocessed_files() == {
    Path(empty_wav1.name).absolute(),
    Path(empty_wav2.name).absolute(),
  }


def test_tflite_fp32_np_array() -> None:
  sf_read = sf.read(TEST_FILE_SHORT)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run_arrays(sf_read)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tflite_fp32_two_np_arrays() -> None:
  sf_read = sf.read(TEST_FILE_SHORT)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run_arrays([sf_read, sf_read])
  assert res.species_probs.shape == (2, 3, 6522)


def xtest_tflite_fp32_large_np_array() -> None:
  sf_data, sr = sf.read(TEST_FILE_LONG, dtype="float32")
  data_6h = numpy.tile(sf_data, 30 * 6)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(top_k=None) as session:
    res = session.run_arrays((data_6h, sr))
  assert res.species_probs.shape == (1, 40 * 30 * 6, 6522)


def test_tflite_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tflite_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tflite_int8_all_species_no_threshold_should_not_mask_anything() -> None:
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


def test_tflite_fp32_twice_two_sessions() -> None:
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


def run_session_process(
  x: multiprocessing.synchronize.Barrier, queue: multiprocessing.Queue
) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  x.wait()
  with model.predict_session(n_workers=1, top_k=None) as session:
    result = session.run(TEST_FILE_SHORT)
  queue.put(result)


def test_tflite_fp32_twice_two_sessions_parallel_processes_fork() -> None:
  use_fork_or_skip()

  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session_process, args=(x, queue))

    p1.start()
    p2.start()

    res1 = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  assert_prediction_result_is_equal(res1, res2)


def test_tflite_fp32_twice_two_sessions_parallel_processes_forkserver() -> None:
  use_forkserver_or_skip()
  with multiprocessing.Manager() as manager:
    x = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(x, queue))
    p2 = multiprocessing.Process(target=run_session_process, args=(x, queue))

    p1.start()
    p2.start()

    res1 = queue.get(timeout=None)
    res2 = queue.get(timeout=None)

    p1.join()
    p2.join()

  assert_prediction_result_is_equal(res1, res2)


def test_tflite_fp32_twice_two_sessions_parallel_processes_spawn() -> None:
  use_spawn_or_skip()
  with multiprocessing.Manager() as manager:
    barrier = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(barrier, queue))
    p2 = multiprocessing.Process(target=run_session_process, args=(barrier, queue))

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
  with model.predict_session(
    n_workers=1, top_k=None, show_stats="benchmark"
  ) as session:
    result = session.run(TEST_FILE_SHORT)
  queue.put(result)


def xtest_tflite_fp32_twice_two_sessions_parallel_threads() -> None:
  # Disabled test because it hangs sometimes, reason unknown.
  n_threads = 10

  barrier = threading.Barrier(n_threads)
  thread_queue = queue.Queue()

  threads: list[threading.Thread] = []
  for i in range(n_threads):
    t = threading.Thread(
      target=run_session_thread,
      args=(barrier, thread_queue),
      name=f"{i}-run_session_thread",
    )
    threads.append(t)

  for t in threads:
    t.start()

  results = []
  for _ in range(n_threads):
    # if run() never finishes, this will timeout and fail the test
    res = thread_queue.get(timeout=None)
    results.append(res)

  for t in threads:
    t.join()

  for i in range(1, n_threads):
    assert_prediction_result_is_equal(results[0], results[i])


def test_tflite_fp32_twice_same_session() -> None:
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
