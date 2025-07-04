
from birdnet.utils import get_chunks_with_overlap


def test_4_2_0__returns_float():
  result = list(get_chunks_with_overlap(4, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
  ]

  assert isinstance(result[0][0], float)
  assert isinstance(result[0][1], float)
  assert isinstance(result[1][0], float)
  assert isinstance(result[1][1], float)


def test_1_2_0__returns_01():
  result = list(get_chunks_with_overlap(1, 2, 0))
  assert result == [
    (0, 1),
  ]


def test_2_2_0__returns_02():
  result = list(get_chunks_with_overlap(2, 2, 0))
  assert result == [
    (0, 2),
  ]


def test_2_2_1__returns_02():
  result = list(get_chunks_with_overlap(2, 2, 1))
  assert result == [
    (0, 2),
  ]


def test_2_2_1p9__returns_02():
  result = list(get_chunks_with_overlap(2, 2, 1.9))
  assert result == [
    (0, 2),
  ]


def test_4_2_0__returns_02_24():
  result = list(get_chunks_with_overlap(4, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
  ]


def test_6_2_0__returns_02_24_46():
  result = list(get_chunks_with_overlap(6, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
    (4, 6),
  ]


def test_6_2_1__returns_02_13_24_35_46():
  result = list(get_chunks_with_overlap(6, 2, 1))
  assert result == [
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
  ]


def test_6_2_0p5__returns_0_2__1p5_3p5__3_5__4p5_6():
  result = list(get_chunks_with_overlap(6, 2, 0.5))
  assert result == [
    (0, 2),
    (1.5, 3.5),
    (3, 5),
    (4.5, 6),
  ]
