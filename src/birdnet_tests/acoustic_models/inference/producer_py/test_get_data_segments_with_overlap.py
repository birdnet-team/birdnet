"""Full segments match the model's input width, whatever the two sample rates."""

import numpy as np
import pytest

from birdnet.acoustic.inference.core.producer import get_data_segments_with_overlap
from birdnet.utils.helper import duration_as_samples

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize(
  ("sample_rate", "segment_duration_s", "target_sample_rate", "speed"),
  [
    # Measured: resampled by the ratio of the rates these come out one sample
    # long (48001, 130479), or one short (79999).
    (11025, 1.5, 32000, 1.0),
    (16000, 2.7183, 48000, 1.0),
    (11025, 2.5, 32000, 1.0),
    (44100, 3.0, 48000, 1.0),
    (22050, 2.5, 32000, 1.5),
  ],
)
def test_full_segments_have_the_model_input_width(
  sample_rate: int, segment_duration_s: float, target_sample_rate: int, speed: float
) -> None:
  width = duration_as_samples(segment_duration_s, target_sample_rate)
  audio = np.random.default_rng(0).standard_normal(sample_rate * 10)
  segments = list(
    get_data_segments_with_overlap(
      audio.astype(np.float32),
      sample_rate,
      segment_duration_s=segment_duration_s,
      overlap_duration_s=0.0,
      speed=speed,
      target_sample_rate=target_sample_rate,
    )
  )
  *full, last = segments
  assert full
  assert [len(s) for s in full] == [width] * len(full)
  assert len(last) <= width
