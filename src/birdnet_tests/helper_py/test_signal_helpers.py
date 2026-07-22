import ctypes
from multiprocessing import Queue

import numpy as np
import pytest

from birdnet.utils.helper import (
  assert_queue_is_empty,
  bandpass_signal,
  check_is_intel_macos,
  uint_ctype_from_dtype,
  xget_max_n_segments,
)


@pytest.mark.parametrize(
  ("dtype", "expected"),
  [
    (np.uint8, ctypes.c_uint8),
    (np.uint16, ctypes.c_uint16),
    (np.uint32, ctypes.c_uint32),
    (np.uint64, ctypes.c_uint64),
  ],
)
def test_uint_ctype_from_dtype(dtype: type, expected: type) -> None:
  assert uint_ctype_from_dtype(dtype) is expected


def test_xget_max_n_segments() -> None:
  # effective segment = 3 - 1 = 2 -> ceil(10 / 2) = 5
  assert xget_max_n_segments(10, 3, 1) == 5
  assert xget_max_n_segments(3, 3, 0) == 1


def test_xget_max_n_segments_rejects_non_positive_effective_duration() -> None:
  with pytest.raises(AssertionError):
    xget_max_n_segments(10, 3, 3)


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


def test_assert_queue_is_empty_passes_for_empty_queue() -> None:
  queue: Queue = Queue()
  # must not raise
  assert_queue_is_empty(queue)


def test_assert_queue_is_empty_raises_for_non_empty_queue() -> None:
  import time

  queue: Queue = Queue()
  queue.put(1)
  # A multiprocessing.Queue flushes items to the underlying pipe on a background
  # feeder thread, so the item only becomes visible after a short, non-
  # deterministic delay. Poll until the leftover item is detected.
  raised = False
  deadline = time.monotonic() + 5.0
  while time.monotonic() < deadline:
    try:
      assert_queue_is_empty(queue)
    except AssertionError:
      raised = True
      break
    time.sleep(0.01)
  assert raised, "assert_queue_is_empty did not detect the queued item"


def test_check_is_intel_macos_false_off_mac(monkeypatch: pytest.MonkeyPatch) -> None:
  import platform

  monkeypatch.setattr(platform, "system", lambda: "Linux")
  assert check_is_intel_macos() is False


def test_check_is_intel_macos_true_on_intel_mac(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import platform

  monkeypatch.setattr(platform, "system", lambda: "Darwin")
  monkeypatch.setattr(platform, "machine", lambda: "x86_64")
  assert check_is_intel_macos() is True


def test_check_is_intel_macos_false_on_arm_mac(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import platform

  monkeypatch.setattr(platform, "system", lambda: "Darwin")
  monkeypatch.setattr(platform, "machine", lambda: "arm64")
  assert check_is_intel_macos() is False
