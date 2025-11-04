import numpy
import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_litert_or_skip,
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_WAV


@pytest.mark.litert
def test_litert_fork() -> None:
  ensure_litert_or_skip()
  use_fork_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf_fork() -> None:
  use_fork_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb_cpu_fork() -> None:
  use_fork_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


@pytest.mark.gpu
def test_pb_gpu_fork() -> None:
  ensure_gpu_or_skip()
  use_fork_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=1, device="GPU")

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


@pytest.mark.litert
def test_litert_forkserver() -> None:
  ensure_litert_or_skip()
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf_forkserver() -> None:
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb_cpu_forkserver() -> None:
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=2, device="CPU")

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


@pytest.mark.gpu
def test_pb_gpu_forkserver() -> None:
  ensure_gpu_or_skip()
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=1, device="GPU")

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


@pytest.mark.litert
def test_litert_spawn() -> None:
  ensure_litert_or_skip()
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf_spawn() -> None:
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb_cpu_spawn() -> None:
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=2, device="CPU")

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


@pytest.mark.gpu
def test_pb_gpu_spawn() -> None:
  ensure_gpu_or_skip()
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=1, device="GPU")

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)
