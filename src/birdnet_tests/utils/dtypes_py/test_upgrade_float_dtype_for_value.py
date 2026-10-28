import numpy as np

from birdnet.utils.dtypes import upgrade_float_dtype_for_value


def test_upgrade_float_dtype_keeps_exactly_representable_value() -> None:
  # 0.5 is exactly representable in float16, so no upgrade happens.
  result = upgrade_float_dtype_for_value(np.dtype(np.float16), 0.5)
  assert result == np.dtype(np.float16)


def test_upgrade_float_dtype_promotes_inexact_value() -> None:
  # 0.1 is not exactly representable in float16 or float32 -> promoted to float64.
  result = upgrade_float_dtype_for_value(np.dtype(np.float16), 0.1)
  assert result == np.dtype(np.float64)
