from birdnet.acoustic_models.inference.producer import get_segments_with_overlap_samples


def test_n_samples_smaller_than_segment() -> None:
  result = list(get_segments_with_overlap_samples(10_000, 15_000, 0, speed=1.0))

  assert result[0][0] == 0
  assert result[-1][1] == 10_000
  assert result == [
    (0, 10_000, 10_000),
  ]


def test_n_samples_smaller_than_segment_with_overlap() -> None:
  result = list(get_segments_with_overlap_samples(10_000, 15_000, 5_000, speed=1.0))

  assert result[0][0] == 0
  assert result[-1][1] == 10_000
  assert result == [
    (0, 10_000, 10_000),
  ]


def test_overlap_larger_n_samples() -> None:
  result = list(get_segments_with_overlap_samples(10_000, 15_000, 14_999, speed=1.0))

  assert result[0][0] == 0
  assert result[-1][1] == 10_000
  assert len(result) == 10_000


def test_smallest_overlap() -> None:
  result = list(get_segments_with_overlap_samples(10_000, 15_000, 1, speed=1.0))

  assert result[0][0] == 0
  assert result[-1][1] == 10_000
  assert result == [
    (0, 10_000, 10_000),
  ]


def test_no_speedup() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=1.0))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 15_000, 15_000),
    (15_000, 30_000, 15_000),
    (30_000, 40_000, 10_000),
  ]


def test_no_speedup_with_overlap() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 5_000, speed=1.0))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 15000, 15000),
    (10000, 25000, 15000),
    (20000, 35000, 15000),
    (30000, 40000, 10000),
  ]


def test_double_time() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=2.0))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 30_000, 15_000),
    (30_000, 40_000, 5_000),
  ]


def test_double_time_with_overlap() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 5_000, speed=2.0))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [(0, 30000, 15000), (20000, 40000, 10000)]


def test_half_time() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=0.5))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 7_500, 15_000),
    (7_500, 15_000, 15_000),
    (15_000, 22_500, 15_000),
    (22_500, 30_000, 15_000),
    (30_000, 37_500, 15_000),
    (37_500, 40_000, 5_000),
  ]


def test_half_time_with_overlap() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 5_000, speed=0.5))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 7500, 15000),
    (5000, 12500, 15000),
    (10000, 17500, 15000),
    (15000, 22500, 15000),
    (20000, 27500, 15000),
    (25000, 32500, 15000),
    (30000, 37500, 15000),
    (35000, 40000, 10000),
  ]


def test_speed_0_1() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=0.1))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 1_500, 15_000),
    (1_500, 3_000, 15_000),
    (3_000, 4_500, 15_000),
    (4_500, 6_000, 15_000),
    (6_000, 7_500, 15_000),
    (7_500, 9_000, 15_000),
    (9_000, 10_500, 15_000),
    (10_500, 12_000, 15_000),
    (12_000, 13_500, 15_000),
    (13_500, 15_000, 15_000),
    (15_000, 16_500, 15_000),
    (16_500, 18_000, 15_000),
    (18_000, 19_500, 15_000),
    (19_500, 21_000, 15_000),
    (21_000, 22_500, 15_000),
    (22_500, 24_000, 15_000),
    (24_000, 25_500, 15_000),
    (25_500, 27_000, 15_000),
    (27_000, 28_500, 15_000),
    (28_500, 30_000, 15_000),
    (30_000, 31_500, 15_000),
    (31_500, 33_000, 15_000),
    (33_000, 34_500, 15_000),
    (34_500, 36_000, 15_000),
    (36_000, 37_500, 15_000),  # span of 1_500 (37_500 - 36_000)
    (37_500, 39_000, 15_000),
    (
      39_000,
      40_000,
      10_000,
    ),  # span of 1_000 instead of 1_500 -> 10_000 instead of 15_000 (2/3)
  ]


def test_speed_10() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=10.0))

  assert result[0][0] == 0
  assert result[-1][1] == 40_000
  assert result == [
    (0, 40_000, 4000),
  ]


def test_odd_samples() -> None:
  result = list(get_segments_with_overlap_samples(41_005, 15_003, 0, speed=2.0))

  assert result[0][0] == 0
  assert result[-1][1] == 41_005
  assert result == [
    (0, 30006, 15003),
    (30006, 41005, 5499),
  ]


def test_floating_speed_round_down_seg_len() -> None:
  # will result in 7000.15 samples per segment -> 7500 after rounding
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=0.50001))

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert result == [
    (0, 7_500, 15_000),
    (7_500, 15_000, 15_000),
    (15_000, 22_500, 15_000),
    (22_500, 30_000, 15_000),
    (30_000, 37_500, 15_000),
    (37_500, 40_000, 4_998),
  ]


def test_floating_speed_round_up_seg_len() -> None:
  # will result in 7500.6 samples per segment -> 7501 after rounding
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=0.50004))

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert result == [
    (0, 7501, 15000),
    (7501, 15002, 15000),
    (15002, 22503, 15000),
    (22503, 30004, 15000),
    (30004, 37505, 15000),
    (37505, 40000, 4994),
  ]


def test_long_floating_speed_has_no_effect_of_n_samples_15000() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=1.12646987))

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert result == [
    (0, 16897, 15000),
    (16897, 33794, 15000),
    (33794, 40000, 5509),
  ]


def test_minimun_speedup() -> None:
  speedup_for_seg_size_1 = 1 / 15_000
  result = list(
    get_segments_with_overlap_samples(40_000, 15_000, 0, speed=speedup_for_seg_size_1)
  )

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert len(result) == 40000


def test_minimun_speedup_with_overlap() -> None:
  speedup_for_seg_size_1 = 2 / 15_000
  overlap_samples = 1 / speedup_for_seg_size_1
  result = list(
    get_segments_with_overlap_samples(
      40_000, 15_000, int(overlap_samples), speed=speedup_for_seg_size_1
    )
  )

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert len(result) == 40000


def test_very_high_speedup() -> None:
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=10_000))

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert result == [
    (0, 40000, 4),
  ]


def test_max_speedup() -> None:
  max_speedup = 40_000
  result = list(get_segments_with_overlap_samples(40_000, 15_000, 0, speed=max_speedup))

  assert result[0][0] == 0
  assert result[-1][1] == 40000
  assert result == [
    (0, 40000, 1),
  ]
