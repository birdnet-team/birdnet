import numpy

from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_WAV


def test_v2_4_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  res = model.encode(TEST_FILE_WAV, n_workers=1)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3406, rtol=1e-4)


def test_v2_4_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  res = model.encode(TEST_FILE_WAV, n_workers=1)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3406, rtol=1e-4)


def test_v2_4_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  res = model.encode(TEST_FILE_WAV, n_workers=1)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3347, rtol=1e-4)
