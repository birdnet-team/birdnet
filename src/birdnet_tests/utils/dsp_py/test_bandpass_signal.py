import numpy as np
import pytest

from birdnet.utils.dsp import bandpass_signal


@pytest.mark.parametrize(
  ("fmin", "fmax", "new_fmin", "new_fmax"),
  [
    (500, 15000, 0, 15000),  # highpass
    (0, 8000, 0, 15000),  # lowpass
    (500, 8000, 0, 15000),  # bandpass
  ],
)
def test_bandpass_signal_filters_and_preserves_shape(
  fmin: int, fmax: int, new_fmin: int, new_fmax: int
) -> None:
  rate = 48000
  signal = np.random.default_rng(0).normal(0, 1, rate).astype(np.float32)
  result = bandpass_signal(signal, rate, fmin, fmax, new_fmin, new_fmax)

  assert result.dtype == np.float32
  assert result.shape == signal.shape
  # a filter was applied, so the signal changed
  assert not np.array_equal(result, signal)


def test_bandpass_signal_no_op_when_range_unchanged() -> None:
  rate = 48000
  signal = np.random.default_rng(1).normal(0, 1, rate).astype(np.float32)
  result = bandpass_signal(signal, rate, 0, 15000, 0, 15000)

  assert result.dtype == np.float32
  np.testing.assert_array_equal(result, signal)
