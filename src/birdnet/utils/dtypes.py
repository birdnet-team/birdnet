"""Narrowest NumPy dtypes for given values, and their ctypes counterparts."""

from __future__ import annotations

import ctypes
from typing import TypeAlias

import numpy as np
from numpy.typing import DTypeLike

UIntCType: TypeAlias = (
  ctypes.c_uint8 | ctypes.c_uint16 | ctypes.c_uint32 | ctypes.c_uint64
)

_UINT_DTYPE_TO_CTYPE: dict[type[np.generic], type[UIntCType]] = {
  np.uint8: ctypes.c_uint8,
  np.uint16: ctypes.c_uint16,
  np.uint32: ctypes.c_uint32,
  np.uint64: ctypes.c_uint64,
}


def max_value_for_uint_dtype(dtype: np.dtype) -> int:
  """
  Returns the maximum value that can be represented by the given NumPy dtype.
  """
  assert np.issubdtype(dtype, np.integer)
  return np.iinfo(dtype).max


def uint_ctype_from_dtype(dtype: DTypeLike) -> type[UIntCType]:
  dtype = np.dtype(dtype).type
  code = _UINT_DTYPE_TO_CTYPE[dtype]
  return code


def get_uint_dtype(max_value: int) -> np.dtype:
  """
  Return the narrowest unsigned-integer NumPy dtype that can represent
  *max_value* (inclusive).

  Examples
  --------
  >>> get_uint_dtype(100)
  dtype('uint8')
  >>> get_uint_dtype(42_000)
  dtype('uint16')
  >>> get_uint_dtype(5_000_000_000)
  dtype('uint64')

  Notes
  -----
  2**8 = 256
  2**16 = 65,536
  2**32 = 4,294,967,296
  2**64 = 18,446,744,073,709,551,616
  """
  assert max_value >= 0, "max_value must be non-negative."

  for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
    if max_value <= np.iinfo(dt).max:
      return np.dtype(dt)

  raise AssertionError("Value exceeds uint64 range.")


def get_float_dtype(max_value: float) -> DTypeLike:
  """
  Magnitude-based: returns the smallest float dtype whose range covers max_value.
  Use for bulk arrays where memory matters and per-element
  rounding is acceptable (e.g. lists of file durations).
  """
  if max_value <= 2**11:
    return np.float16
  elif max_value <= 2**24:
    return np.float32
  else:
    return np.float64


def upgrade_float_dtype_for_value(dtype: np.dtype, value: float) -> np.dtype:
  if dtype == np.float16 and float(np.float16(value)) != float(value):
    dtype = np.dtype(np.float32)
  if dtype == np.float32 and float(np.float32(value)) != float(value):
    dtype = np.dtype(np.float64)
  return dtype


# Lossless: smallest float dtype that represents value exactly. Use for scalar
# configuration parameters (speed, segment/overlap duration) where the value
# feeds into derived computations and rounding accumulates over many segments.
def get_lossless_float_dtype(value: float) -> np.dtype:
  return upgrade_float_dtype_for_value(np.dtype(get_float_dtype(value)), value)
