
import numpy
import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILE_WAV


@pytest.mark.repro
def test_pb_cpu_fp32() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  res = model.predict(TEST_FILE_WAV, n_workers=4, device="CPU")
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.062329773)


@pytest.mark.repro
@pytest.mark.gpu
def test_pb_gpu_fp32() -> None:
  ensure_gpu_or_skip()
    
  model = load("acoustic", "2.4", "pb", precision="fp32")
  res = model.predict(TEST_FILE_WAV, n_workers=1, device="GPU")
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  # last decimal differs on different runs
  numpy.testing.assert_almost_equal(mean, 0.0623320, decimal=7)


@pytest.mark.repro
def test_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=4)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.062330008)


@pytest.mark.repro
def test_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=4)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.06250076)


@pytest.mark.repro
def test_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  with model.predict_session(n_workers=4) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.06406734)


@pytest.mark.repro
@pytest.mark.litert
def test_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  with model.predict_session(n_workers=4) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.06233002)


@pytest.mark.repro
@pytest.mark.litert
def test_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  with model.predict_session(n_workers=4) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.062500775)


@pytest.mark.repro
@pytest.mark.litert
def test_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  with model.predict_session(n_workers=4) as session:
    res = session.run(TEST_FILE_WAV)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_equal(mean, 0.06300629)
