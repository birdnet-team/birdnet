from collections.abc import Generator
from itertools import count

import numpy.testing

# def get_segments_with_overlap(
#   total_duration_s: int | float,
#   segment_duration_s: int | float,
#   overlap_duration_s: int | float,
# ) -> Generator[tuple[float, float], None, None]:
#   assert total_duration_s > 0
#   assert segment_duration_s > 0
#   assert 0 <= overlap_duration_s < segment_duration_s

#   if not isinstance(overlap_duration_s, float):
#     overlap_duration_s = float(overlap_duration_s)
#   if not isinstance(segment_duration_s, float):
#     segment_duration_s = float(segment_duration_s)
#   if not isinstance(total_duration_s, float):
#     total_duration_s = float(total_duration_s)

#   step_duration = segment_duration_s - overlap_duration_s
#   for start in count(0.0, step=step_duration):
#     assert start < total_duration_s
#     if (end := start + segment_duration_s) < total_duration_s:
#       yield start, end
#     else:
#       yield start, total_duration_s
#       break


def get_segments_with_overlap_all(
  total_duration_s: int | float,
  segment_duration_s: int | float,
  overlap_duration_s: int | float,
) -> Generator[tuple[float, float], None, None]:
  assert total_duration_s > 0
  assert segment_duration_s > 0
  assert 0 <= overlap_duration_s < segment_duration_s

  if not isinstance(overlap_duration_s, float):
    overlap_duration_s = float(overlap_duration_s)
  if not isinstance(segment_duration_s, float):
    segment_duration_s = float(segment_duration_s)
  if not isinstance(total_duration_s, float):
    total_duration_s = float(total_duration_s)

  step_duration = segment_duration_s - overlap_duration_s
  for start in count(0.0, step=step_duration):
    if start >= total_duration_s:
      break
    end = min(start + segment_duration_s, total_duration_s)
    yield start, end


def test_1_2_0__returns_01() -> None:
  result = list(get_segments_with_overlap_all(1, 2, 0))
  assert result == [
    (0, 1),
  ]


def test_2_2_0__returns_02() -> None:
  result = list(get_segments_with_overlap_all(2, 2, 0))
  assert result == [
    (0, 2),
  ]


def test_2_2_1__returns_02() -> None:
  result = list(get_segments_with_overlap_all(2, 2, 1))
  assert result == [
    (0, 2),
    (1, 2),  # last segment shorter because it reaches the end
  ]


def test_2_2_1p9__returns_02() -> None:
  result = list(get_segments_with_overlap_all(2, 2, 1.9))
  assert_result = [(0.1 * i, 2) for i in range(20)]
  numpy.testing.assert_allclose(result, assert_result, rtol=1e-8)


def test_4_2_0__returns_02_24() -> None:
  result = list(get_segments_with_overlap_all(4, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
  ]


def test_int_input__returns_float() -> None:
  result = list(get_segments_with_overlap_all(4, 2, 0))

  assert len(result) == 2
  assert isinstance(result[0][0], float)
  assert isinstance(result[0][1], float)
  assert isinstance(result[1][0], float)
  assert isinstance(result[1][1], float)


def test_float_input__returns_float() -> None:
  result = list(get_segments_with_overlap_all(4.0, 2.0, 0.0))

  assert len(result) == 2
  assert isinstance(result[0][0], float)
  assert isinstance(result[0][1], float)
  assert isinstance(result[1][0], float)
  assert isinstance(result[1][1], float)


def test_6_2_0__returns_02_24_46() -> None:
  result = list(get_segments_with_overlap_all(6, 2, 0))
  assert result == [
    (0, 2),
    (2, 4),
    (4, 6),
  ]


def test_6_2_1__returns_02_13_24_35_46() -> None:
  result = list(get_segments_with_overlap_all(6, 2, 1))
  assert result == [
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),  # last segment shorter because it reaches the end
  ]


def test_6_2_0p5__returns_0_2__1p5_3p5__3_5__4p5_6() -> None:
  result = list(get_segments_with_overlap_all(6, 2, 0.5))
  assert result == [
    (0, 2),
    (1.5, 3.5),
    (3, 5),
    (4.5, 6),
  ]
