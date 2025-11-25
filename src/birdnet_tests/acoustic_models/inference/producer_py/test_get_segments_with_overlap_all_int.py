
from birdnet.acoustic_models.inference.producer import get_segments_with_overlap_all_int


def test_1_2_0__returns_01() -> None:
  result = list(get_segments_with_overlap_all_int(1, 2, 0))
  assert result == [
    (0, 1),
  ]


def test_2_2_0__returns_02() -> None:
  result = list(get_segments_with_overlap_all_int(2, 2, 0))
  assert result == [
    (0, 2),
  ]


def test_2_2_1__returns_02() -> None:
  result = list(get_segments_with_overlap_all_int(2, 2, 1))
  assert result == [
    (0, 2),
    (1, 2),  # last segment shorter because it reaches the end
  ]


def test_4_2_0__returns_02_24() -> None:
  result = list(get_segments_with_overlap_all_int(4, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
  ]


def test_6_2_0__returns_02_24_46() -> None:
  result = list(get_segments_with_overlap_all_int(6, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
    (4, 6),
  ]


def test_6_2_1__returns_02_13_24_35_46() -> None:
  result = list(get_segments_with_overlap_all_int(6, 2, 1))
  assert result == [
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),  # last segment shorter because it reaches the end
  ]
