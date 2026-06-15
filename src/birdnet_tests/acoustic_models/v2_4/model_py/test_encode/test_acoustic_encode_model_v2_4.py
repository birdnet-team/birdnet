import pytest
import soundfile

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILE_SHORT, TEST_FILE_SHORT_EMB_SHAPE


def test_tflite_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_tflite_fp32_np_array() -> None:
  sf_read = soundfile.read(TEST_FILE_SHORT)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.encode_session(n_workers=1) as session:
    res = session.run_arrays(sf_read)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_tflite_fp32_two_np_arrays() -> None:
  sf_read = soundfile.read(TEST_FILE_SHORT)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.encode_session(n_workers=1) as session:
    res = session.run_arrays([sf_read, sf_read])
  assert res.embeddings.shape == (2, 3, 1024)


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
