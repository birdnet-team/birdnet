
import numpy

from birdnet.model_loader import load
from birdnet_tests.helper import (
  use_fork_or_skip,
  use_forkserver_or_skip,
  use_spawn_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_WAV


def test_litert_fork() -> None:
  use_fork_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf_fork() -> None:
  use_fork_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb_fork() -> None:
  use_fork_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_litert_forkserver() -> None:
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf_forkserver() -> None:
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb_forkserver() -> None:
  use_forkserver_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_litert_spawn() -> None:
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_tf_spawn() -> None:
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)


def test_pb_spawn() -> None:
  use_spawn_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  result = model.predict(TEST_FILE_WAV, n_workers=2)

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.06287, decimal=5)
