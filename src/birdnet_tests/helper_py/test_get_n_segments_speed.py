from birdnet.acoustic_models.inference.producer import get_segments_with_overlap_samples
from birdnet.helper import duration_as_samples, get_n_segments_speed


def assert_equal_to_get_segments_fn(
  max_duration_s: float, segment_size_s: float, overlap_duration_s: float, speed: float
) -> None:
  n_segments = get_n_segments_speed(
    max_duration_s, segment_size_s, overlap_duration_s, speed
  )
  sr = 48_000
  assert_n_segments = len(
    list(
      get_segments_with_overlap_samples(
        duration_as_samples(max_duration_s, sr),
        duration_as_samples(segment_size_s, sr),
        duration_as_samples(overlap_duration_s, sr),
        speed,
      )
    )
  )

  assert n_segments == assert_n_segments


def test_speed_1() -> None:
  n_seg = get_n_segments_speed(120, 3, 0, 1.0)
  assert n_seg == 40


def test_speed_doubletime() -> None:
  n_seg = get_n_segments_speed(120, 3, 0, 2.0)
  assert n_seg == 20


def test_speed_halftime() -> None:
  n_seg = get_n_segments_speed(120, 3, 0, 0.5)
  assert n_seg == 80


def test_speed_with_overlap() -> None:
  max_duration_s = 120
  segment_size_s = 3
  overlap_duration_s = 1
  speed = 2.0

  assert_equal_to_get_segments_fn(
    max_duration_s, segment_size_s, overlap_duration_s, speed
  )


def test_120_4_2_3() -> None:
  max_duration_s = 120
  segment_size_s = 4
  overlap_duration_s = 2
  speed = 3.0

  assert_equal_to_get_segments_fn(
    max_duration_s, segment_size_s, overlap_duration_s, speed
  )


def test_120_4_0_3() -> None:
  max_duration_s = 120
  segment_size_s = 4
  overlap_duration_s = 0
  speed = 3.0

  assert_equal_to_get_segments_fn(
    max_duration_s, segment_size_s, overlap_duration_s, speed
  )


def test_120_4_29_123465() -> None:
  max_duration_s = 120
  segment_size_s = 4
  overlap_duration_s = 2.9
  speed = 1.23465

  assert_equal_to_get_segments_fn(
    max_duration_s, segment_size_s, overlap_duration_s, speed
  )


def test_7_3_1_05() -> None:
  max_duration_s = 7
  segment_size_s = 3
  overlap_duration_s = 1
  speed = 0.5

  assert_equal_to_get_segments_fn(
    max_duration_s, segment_size_s, overlap_duration_s, speed
  )
