import multiprocessing
import multiprocessing.synchronize
from pathlib import Path

import numpy
import pytest
import soundfile as sf

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.model_loader import load
from birdnet_tests.helper import (
  assert_prediction_result_is_close,
  assert_prediction_result_is_equal,
  create_zero_len_wav,
  ensure_gpu_or_skip,
  ensure_litert_or_skip,
  join_children_or_fail,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import (
  TEST_FILE_SHORT,
  TEST_FILE_SHORT_SCORE_SHAPE,
)

# Two independent TFLite/PB interpreter instances are not guaranteed to produce
# bit-identical output for the same input: XNNPACK sizes its thread pool from
# the cores it can see, and a different thread count changes the reduction
# order. Comparisons that span sessions, processes or threads therefore use a
# tolerance; a comparison served by a single interpreter stays exact.
_CROSS_INSTANCE_MAX_ABS_DIFF = 1e-6


def _in_species_order(
  res: AcousticPredictionResultBase,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
  """ids, scores and masks with the species axis sorted, as top-k may permute it."""
  order = numpy.argsort(res.species_ids, axis=-1)
  return (
    numpy.take_along_axis(res.species_ids, order, axis=-1),
    numpy.take_along_axis(res.species_probs, order, axis=-1),
    numpy.take_along_axis(res.species_masked, order, axis=-1),
  )


def test_pb_cpu_fp32() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None, device="CPU") as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None, device="CPU") as session:
    res3 = session.run(TEST_FILE_SHORT)

  assert res1.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE
  assert_prediction_result_is_equal(res1, res2)
  assert_prediction_result_is_close(
    res1, res3, max_abs_diff=_CROSS_INSTANCE_MAX_ABS_DIFF
  )


@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.predict_session(n_workers=1, top_k=None, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE


def test_tflite_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  array = sf.read(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None) as session:
    from_file = session.run(TEST_FILE_SHORT)
    from_file_again = session.run(TEST_FILE_SHORT)
    from_array = session.run_arrays(array)
    from_two_arrays = session.run_arrays([array, array])
  with model.predict_session(n_workers=1, top_k=None) as session:
    from_second_session = session.run(TEST_FILE_SHORT)

  assert from_file.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE
  assert_prediction_result_is_equal(from_file, from_file_again)
  assert from_array.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE
  assert from_two_arrays.species_probs.shape == (2, 3, 6522)

  # One interpreter: the two copies of the array come out exactly alike.
  ids, probs, masks = _in_species_order(from_two_arrays)
  numpy.testing.assert_array_equal(ids[0], ids[1])
  numpy.testing.assert_array_equal(probs[0], probs[1])
  numpy.testing.assert_array_equal(masks[0], masks[1])

  # The same bytes, read as float32 by the producer and as float64 by soundfile,
  # through the same per-segment stretch (measured: identical).
  file_ids, file_probs, file_masks = _in_species_order(from_file)
  array_ids, array_probs, array_masks = _in_species_order(from_array)
  numpy.testing.assert_array_equal(file_ids, array_ids)
  numpy.testing.assert_array_equal(file_masks, array_masks)
  numpy.testing.assert_allclose(
    file_probs, array_probs, rtol=0, atol=_CROSS_INSTANCE_MAX_ABS_DIFF
  )

  assert_prediction_result_is_close(
    from_file, from_second_session, max_abs_diff=_CROSS_INSTANCE_MAX_ABS_DIFF
  )


def test_tflite_fp32_speed_factor() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1, top_k=None, speed=0.5, default_confidence_threshold=-numpy.inf
  ) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.species_probs.shape == (1, 5, 6522)
  # The shape is the session's allocation; only an unmasked row shows the
  # producer wrote that segment.
  assert not res.species_masked.any()


def test_tflite_fp32_unreadable_inputs(tmp_path: Path) -> None:
  empty_a = tmp_path / "empty_a.wav"
  empty_a.write_bytes(b"")
  zero_frames = tmp_path / "zero_frames.wav"
  with zero_frames.open("wb") as f:
    create_zero_len_wav(f)
  broken = tmp_path / "broken.wav"
  broken.write_bytes(b"NOT_A_VALID_WAV_FILE")
  empty_b = tmp_path / "empty_b.wav"
  empty_b.write_bytes(b"")
  inputs = [empty_a, TEST_FILE_SHORT, zero_frames, broken, empty_b]
  unreadable = {empty_a, broken, empty_b}

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  # Two producers pull from one queue; which one reads which input depends on
  # timing, so the union of their reports is pinned by a unit test, not here.
  with model.predict_session(
    n_workers=1, n_producers=2, top_k=None, default_confidence_threshold=-numpy.inf
  ) as session:
    res = session.run(inputs)

  assert res.species_probs.shape == (5, 3, 6522)
  assert res.get_unprocessed_files() == {p.absolute() for p in unreadable}
  # Rows follow the result's input order (sorted), not the call's.
  row = {Path(p): i for i, p in enumerate(res.inputs)}
  unreadable_rows = {row[p.absolute()] for p in unreadable}
  assert set(res.unprocessable_inputs.tolist()) == unreadable_rows
  for i in unreadable_rows:
    assert numpy.all(res.species_probs[i] == 0)
    assert numpy.all(res.species_ids[i] == 0)
    assert numpy.all(res.species_masked[i])
  # Readable but without a frame: not unprocessable, and no segment is written,
  # so the row is padding: masked, its values never defined (tensor._allocate).
  assert numpy.all(res.species_masked[row[zero_frames.absolute()]])
  valid_row = row[TEST_FILE_SHORT.absolute()]
  assert numpy.all(res.species_probs[valid_row] != 0)
  assert not numpy.any(res.species_masked[valid_row])


def test_tflite_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
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
  assert res.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE
  assert numpy.all(~res.species_masked)


def test_tflite_fp32_softmax_matches_softmax_of_logits() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-numpy.inf,
    apply_sigmoid=False,
  ) as session:
    res_logits = session.run(TEST_FILE_SHORT)
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-numpy.inf,
    apply_sigmoid=False,
    apply_softmax=True,
  ) as session:
    res_softmax = session.run(TEST_FILE_SHORT)

  # top-k order is random, so sort both results by species id before comparing
  logits = numpy.take_along_axis(
    res_logits.species_probs, numpy.argsort(res_logits.species_ids, axis=-1), axis=-1
  )
  softmax_probs = numpy.take_along_axis(
    res_softmax.species_probs, numpy.argsort(res_softmax.species_ids, axis=-1), axis=-1
  )

  exp_logits = numpy.exp(logits - logits.max(axis=-1, keepdims=True))
  expected = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
  numpy.testing.assert_allclose(softmax_probs, expected, atol=1e-6)


@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=1, top_k=None) as session:
    res1 = session.run(TEST_FILE_SHORT)
    res2 = session.run(TEST_FILE_SHORT)
  with model.predict_session(n_workers=1, top_k=None) as session:
    res3 = session.run(TEST_FILE_SHORT)

  assert res1.species_probs.shape == TEST_FILE_SHORT_SCORE_SHAPE
  assert_prediction_result_is_equal(res1, res2)
  assert_prediction_result_is_close(
    res1, res3, max_abs_diff=_CROSS_INSTANCE_MAX_ABS_DIFF
  )


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


def run_session_process(
  x: multiprocessing.synchronize.Barrier, queue: multiprocessing.Queue
) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  x.wait()
  with model.predict_session(n_workers=1, top_k=None) as session:
    result = session.run(TEST_FILE_SHORT)
  queue.put(result)


def test_tflite_fp32_twice_two_sessions_parallel_processes_spawn() -> None:
  use_spawn_or_skip()
  with multiprocessing.Manager() as manager:
    barrier = manager.Barrier(2)
    queue = manager.Queue()

    p1 = multiprocessing.Process(target=run_session_process, args=(barrier, queue))
    p2 = multiprocessing.Process(target=run_session_process, args=(barrier, queue))

    p1.start()
    p2.start()

    join_children_or_fail(p1, p2)
    res1 = queue.get(timeout=10)
    res2 = queue.get(timeout=10)

  assert_prediction_result_is_close(
    res1, res2, max_abs_diff=_CROSS_INSTANCE_MAX_ABS_DIFF
  )


@pytest.mark.gpu
@pytest.mark.usefixtures("deterministic_tf_ops")
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
