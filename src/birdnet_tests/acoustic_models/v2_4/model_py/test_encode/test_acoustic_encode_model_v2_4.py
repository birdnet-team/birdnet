import numpy
import pytest
import soundfile

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILE_SHORT, TEST_FILE_SHORT_EMB_SHAPE


def test_tflite_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  array = soundfile.read(TEST_FILE_SHORT)
  with model.encode_session(n_workers=1) as session:
    from_file = session.run(TEST_FILE_SHORT)
    from_array = session.run_arrays(array)
    from_two_arrays = session.run_arrays([array, array])

  assert from_file.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE
  assert from_array.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE
  assert from_two_arrays.embeddings.shape == (2, 3, 1024)
  # One interpreter: the two copies of the array come out exactly alike, and
  # the same bytes read as float32 by the producer and as float64 by soundfile
  # come out as the file does (measured: identical).
  numpy.testing.assert_array_equal(
    from_two_arrays.embeddings[0], from_two_arrays.embeddings[1]
  )
  numpy.testing.assert_allclose(
    from_array.embeddings, from_file.embeddings, rtol=0, atol=1e-6
  )
  numpy.testing.assert_allclose(
    from_two_arrays.embeddings[0], from_file.embeddings[0], rtol=0, atol=1e-6
  )


def test_tflite_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_tflite_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


@pytest.mark.litert
def test_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


@pytest.mark.litert
def test_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_v2_4_pb_fp32_cpu() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.encode_session(n_workers=1, device="CPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


@pytest.mark.gpu
def test_pb_gpu() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  with model.encode_session(n_workers=1, device="GPU") as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE
