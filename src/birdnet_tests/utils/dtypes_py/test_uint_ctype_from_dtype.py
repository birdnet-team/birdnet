import ctypes

import numpy as np
import pytest

from birdnet.utils.dtypes import uint_ctype_from_dtype


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
