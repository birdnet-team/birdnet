import numpy as np

from birdnet.utils.dsp import flat_sigmoid_logaddexp_fast


def test_flat_sigmoid_logaddexp_fast_zero_is_half() -> None:
  result = flat_sigmoid_logaddexp_fast(
    np.array([0.0], dtype=np.float32), sensitivity=1.0
  )
  np.testing.assert_allclose(result, [0.5], atol=1e-6)


def test_flat_sigmoid_logaddexp_fast_is_bounded_and_monotonic() -> None:
  # The prediction worker calls this with sensitivity=-1.0, which yields the
  # standard increasing sigmoid; clipping keeps extreme logits finite.
  x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float32)
  result = flat_sigmoid_logaddexp_fast(x, sensitivity=-1.0)
  assert np.all(result >= 0.0)
  assert np.all(result <= 1.0)
  assert np.all(np.isfinite(result))
  # sigmoid is monotonically increasing in the logit
  assert np.all(np.diff(result) >= 0)


def test_flat_sigmoid_logaddexp_fast_matches_reference_sigmoid() -> None:
  # With sensitivity=-1 and bias=1 (transformed_bias=0) it is 1 / (1 + e^-x).
  x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
  result = flat_sigmoid_logaddexp_fast(x, sensitivity=-1.0)
  expected = 1.0 / (1.0 + np.exp(-x))
  np.testing.assert_allclose(result, expected, atol=1e-9)
