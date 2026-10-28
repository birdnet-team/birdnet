import numpy as np
import pytest

from birdnet.utils.dtypes import get_uint_dtype


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
