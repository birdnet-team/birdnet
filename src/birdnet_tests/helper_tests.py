import numpy as np

from birdnet_tests.helper import worst_decimal_precision


def test_worst_decimal_precision_component():
  a = np.array([0.12345678, 0.5, 0.003])
  b = np.array([0.12345670, 0.5, 0.0012])  # diff = 8e-08 → matches 7 decimals

  result = worst_decimal_precision(a, b, max_decimals=8)

  expected = 2
  if result != expected:
    raise Exception(f"Expected precision {expected}, got {result}")


def test_worst_decimal_precision_basic():
  a = np.array([0.12345678, 0.5])
  b = np.array([0.12345670, 0.5])  # diff = 8e-08 → matches 7 decimals

  result = worst_decimal_precision(a, b, max_decimals=8)

  expected = 7
  if result != expected:
    raise Exception(f"Expected precision {expected}, got {result}")


def test_worst_decimal_precision_medium_error():
  a = np.array([0.12])
  b = np.array([0.123])  # diff = 0.003 → matches only 2 decimals

  result = worst_decimal_precision(a, b, max_decimals=8)

  expected = 2
  if result != expected:
    raise Exception(f"Expected precision {expected}, got {result}")


def test_worst_decimal_precision_zero_match():
  a = np.array([1.0])
  b = np.array([1.2])  # diff = 0.2 → matches 0 decimals

  result = worst_decimal_precision(a, b, max_decimals=8)

  expected = 0
  if result != expected:
    raise Exception(f"Expected precision {expected}, got {result}")
