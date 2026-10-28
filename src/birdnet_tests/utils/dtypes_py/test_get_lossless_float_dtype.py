import numpy as np

from birdnet.utils.dtypes import get_lossless_float_dtype


def test_get_lossless_float_dtype() -> None:
  assert get_lossless_float_dtype(0.5) == np.dtype(np.float16)
  assert get_lossless_float_dtype(0.1) == np.dtype(np.float64)
