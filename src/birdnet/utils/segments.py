"""Segment arithmetic: hops, segment counts and sample counts under a speed."""

from __future__ import annotations

import math


def apply_speed_to_duration(duration_s: float, speed: float) -> float:
  assert speed > 0
  scaled_duration = duration_s * speed
  return scaled_duration


def apply_speed_to_samples(samples: int, speed: float) -> int:
  assert speed > 0
  scaled_samples = round(samples * speed)
  return scaled_samples


def get_hop_duration_s(
  segment_size_s: float, overlap_duration_s: float, speed: float
) -> float:
  assert speed > 0
  assert segment_size_s > overlap_duration_s
  hop_duration_s = apply_speed_to_duration(segment_size_s - overlap_duration_s, speed)
  assert hop_duration_s > 0
  return hop_duration_s


def get_n_segments_speed(
  duration_s: float, segment_size_s: float, overlap_duration_s: float, speed: float
) -> int:
  hop_duration_s = get_hop_duration_s(segment_size_s, overlap_duration_s, speed)
  n_segments = math.ceil(duration_s / hop_duration_s)
  return n_segments


def duration_as_samples(duration_s: float, sample_rate: int) -> int:
  return round(duration_s * sample_rate)
