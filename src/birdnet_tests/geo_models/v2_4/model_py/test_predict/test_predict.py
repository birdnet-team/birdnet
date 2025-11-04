import numpy.testing
import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_litert_or_skip


@pytest.mark.litert
def test_litert():
  ensure_litert_or_skip()

  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=False,
  )

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.030548334, decimal=8)


def test_tf():
  model = load("geo", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=False,
  )

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.030548334, decimal=8)


def test_pb():
  model = load("geo", "2.4", "pb", precision="fp32")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=False,
  )

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.030548334, decimal=8)
