import numpy as np
import pytest

from birdnet.utils.helper import (
  get_float_dtype,
  get_lossless_float_dtype,
  get_uint_dtype,
  max_value_for_uint_dtype,
  uint_dtype_for_files,
  upgrade_float_dtype_for_value,
)


@pytest.mark.parametrize(
  ("max_value", "expected"),
  [
    (0, np.uint8),
    (255, np.uint8),
    (256, np.uint16),
    (42_000, np.uint16),
    (65_535, np.uint16),
    (65_536, np.uint32),
    (3_000_000_000, np.uint32),
    (4_294_967_295, np.uint32),
    (4_294_967_296, np.uint64),
    (5_000_000_000_000_000_000, np.uint64),
  ],
)
def test_get_uint_dtype(max_value: int, expected: type) -> None:
  assert get_uint_dtype(max_value) == np.dtype(expected)


def test_get_uint_dtype_negative_raises_error() -> None:
  with pytest.raises(AssertionError, match=r"max_value must be non-negative"):
    get_uint_dtype(-1)


def test_get_uint_dtype_above_uint64_raises_error() -> None:
  with pytest.raises(AssertionError, match=r"Value exceeds uint64 range"):
    get_uint_dtype(2**64)


def test_uint_dtype_for_files_uses_max_index() -> None:
  # 256 files -> indices 0..255 -> fit into uint8
  assert uint_dtype_for_files(256) == np.dtype(np.uint8)
  # 257 files -> index 256 needs uint16
  assert uint_dtype_for_files(257) == np.dtype(np.uint16)


@pytest.mark.parametrize(
  ("max_value", "expected"),
  [
    (0, np.float16),
    (2**11, np.float16),
    (2**11 + 1, np.float32),
    (2**24, np.float32),
    (2**24 + 1, np.float64),
  ],
)
def test_get_float_dtype(max_value: float, expected: type) -> None:
  assert get_float_dtype(max_value) == expected


def test_upgrade_float_dtype_keeps_exactly_representable_value() -> None:
  # 0.5 is exactly representable in float16, so no upgrade happens.
  result = upgrade_float_dtype_for_value(np.dtype(np.float16), 0.5)
  assert result == np.dtype(np.float16)


def test_upgrade_float_dtype_promotes_inexact_value() -> None:
  # 0.1 is not exactly representable in float16 or float32 -> promoted to float64.
  result = upgrade_float_dtype_for_value(np.dtype(np.float16), 0.1)
  assert result == np.dtype(np.float64)


def test_get_lossless_float_dtype() -> None:
  assert get_lossless_float_dtype(0.5) == np.dtype(np.float16)
  assert get_lossless_float_dtype(0.1) == np.dtype(np.float64)


def test_max_value_for_uint_dtype() -> None:
  assert max_value_for_uint_dtype(np.dtype(np.uint8)) == 255
  assert max_value_for_uint_dtype(np.dtype(np.uint16)) == 65_535


def test_max_value_for_uint_dtype_rejects_non_integer() -> None:
  with pytest.raises(AssertionError):
    max_value_for_uint_dtype(np.dtype(np.float32))
