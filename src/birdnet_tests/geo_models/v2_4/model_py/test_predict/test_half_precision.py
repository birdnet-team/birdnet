import numpy.testing

from birdnet.model_loader import load


def test_litert_half():
  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=True,
  )

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.03055, decimal=5)


def test_tf_half():
  model = load("geo", "2.4", "tf", precision="fp32", library="tf")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=True,
  )

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.03055, decimal=5)


def test_pb_half():
  model = load("geo", "2.4", "pb", precision="fp32")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=True,
  )

  numpy.testing.assert_almost_equal(result.species_probs.mean(), 0.03055, decimal=5)
