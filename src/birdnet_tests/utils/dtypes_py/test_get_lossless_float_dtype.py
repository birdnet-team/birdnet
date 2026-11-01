import numpy as np
import pytest

from birdnet.utils.dtypes import get_lossless_float_dtype

pytestmark = pytest.mark.no_tf


def test_get_lossless_float_dtype() -> None:
  assert get_lossless_float_dtype(0.5) == np.dtype(np.float16)
  assert get_lossless_float_dtype(0.1) == np.dtype(np.float64)
