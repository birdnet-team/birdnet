import numpy as np
import pytest

from birdnet.utils.dtypes import max_value_for_uint_dtype


def test_max_value_for_uint_dtype() -> None:
  assert max_value_for_uint_dtype(np.dtype(np.uint8)) == 255
  assert max_value_for_uint_dtype(np.dtype(np.uint16)) == 65_535


def test_max_value_for_uint_dtype_rejects_non_integer() -> None:
  with pytest.raises(AssertionError):
    max_value_for_uint_dtype(np.dtype(np.float32))
