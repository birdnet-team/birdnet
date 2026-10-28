import numpy as np
import pytest

from birdnet.utils.dsp import fillup_with_silence


def test_fillup_with_silence_pads() -> None:
  segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  result = fillup_with_silence(segment, 5)
  np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 0.0, 0.0])
  assert result.dtype == np.float32


def test_fillup_with_silence_returns_same_when_full() -> None:
  segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  result = fillup_with_silence(segment, 3)
  assert result is segment


def test_fillup_with_silence_rejects_too_short_target() -> None:
  segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  with pytest.raises(AssertionError):
    fillup_with_silence(segment, 2)
