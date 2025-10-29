import numpy

from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_WAV


def test_v2_4_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=1)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06287, decimal=4)


def test_v2_4_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=1)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06305, decimal=4)


def test_v2_4_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  res = model.predict(TEST_FILE_WAV, n_workers=1)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 5)
  numpy.testing.assert_almost_equal(mean, 0.06216, decimal=4)
