import numpy as np
import pytest

from birdnet.utils.dtypes import get_float_dtype


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
