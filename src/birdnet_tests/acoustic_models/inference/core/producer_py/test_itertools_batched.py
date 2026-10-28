import pytest

from birdnet.acoustic.inference.core.producer import itertools_batched


def test_itertools_batched() -> None:
  assert list(itertools_batched("ABCDEFG", 3)) == [
    ("A", "B", "C"),
    ("D", "E", "F"),
    ("G",),
  ]


def test_itertools_batched_empty() -> None:
  assert list(itertools_batched([], 3)) == []


def test_itertools_batched_rejects_non_positive_n() -> None:
  with pytest.raises(ValueError, match=r"n must be at least one"):
    list(itertools_batched("ABC", 0))
