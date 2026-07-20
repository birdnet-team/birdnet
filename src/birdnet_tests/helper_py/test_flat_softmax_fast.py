import numpy as np

from birdnet.utils.helper import flat_softmax_fast


def test_rows_sum_to_one() -> None:
  x = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]], dtype=np.float32)

  result = flat_softmax_fast(x)

  np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_values_are_in_zero_one_range() -> None:
  rng = np.random.default_rng(1234)
  x = rng.normal(0.0, 10.0, size=(16, 100)).astype(np.float32)

  result = flat_softmax_fast(x)

  assert np.all(result > 0)
  assert np.all(result <= 1)


def test_equals_naive_softmax() -> None:
  x = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]], dtype=np.float64)
  expected = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

  result = flat_softmax_fast(x)

  np.testing.assert_allclose(result, expected, atol=1e-12)


def test_large_logits_do_not_overflow() -> None:
  # without subtracting the row maximum, exp() would overflow to inf here
  x = np.array([[1000.0, 999.0, 998.0], [-1000.0, -999.0, -998.0]], dtype=np.float32)

  result = flat_softmax_fast(x)

  assert np.all(np.isfinite(result))
  np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_is_shift_invariant() -> None:
  x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

  result = flat_softmax_fast(x)
  result_shifted = flat_softmax_fast(x + 100.0)

  np.testing.assert_allclose(result, result_shifted, atol=1e-6)


def test_preserves_ranking_of_logits() -> None:
  rng = np.random.default_rng(5678)
  x = rng.normal(0.0, 5.0, size=(4, 50)).astype(np.float32)

  result = flat_softmax_fast(x)

  np.testing.assert_array_equal(np.argsort(result, axis=1), np.argsort(x, axis=1))


def test_rows_are_normalized_independently() -> None:
  x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)

  result = flat_softmax_fast(x)
  first_row_alone = flat_softmax_fast(x[:1])

  np.testing.assert_allclose(result[0], first_row_alone[0], atol=1e-6)


def test_single_class_returns_ones() -> None:
  x = np.array([[-3.0], [7.0]], dtype=np.float32)

  result = flat_softmax_fast(x)

  np.testing.assert_allclose(result, [[1.0], [1.0]], atol=1e-6)


def test_equal_logits_return_uniform_distribution() -> None:
  x = np.full((2, 4), 2.5, dtype=np.float32)

  result = flat_softmax_fast(x)

  np.testing.assert_allclose(result, np.full((2, 4), 0.25), atol=1e-6)


def test_preserves_dtype() -> None:
  x_32 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
  x_64 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

  assert flat_softmax_fast(x_32).dtype == np.float32
  assert flat_softmax_fast(x_64).dtype == np.float64
