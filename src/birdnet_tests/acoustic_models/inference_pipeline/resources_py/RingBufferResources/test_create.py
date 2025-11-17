import numpy as np

from birdnet.acoustic_models.inference_pipeline.resources import (
  RingBufferResources,
)
from birdnet.base import get_session_id


def test_ring_names_have_max_30_chars() -> None:
  result = RingBufferResources._create(
    get_session_id(), 4, 1, 48_000 * 3, np.dtype(np.uint32), 1000
  )
  # on macOS the limit is 31 including "/" at the start
  # e.g., this was too long:
  # /bn_ring_file_indices_53879_8340719616_1763049803980550000
  # /bn_ring_file_indices_53879_834 <- this is maximum
  allowed_max_length = 30

  assert len(result.rf_audio_samples.name) <= allowed_max_length
  assert len(result.rf_segment_indices.name) <= allowed_max_length
  assert len(result.rf_file_indices.name) <= allowed_max_length
  assert len(result.rf_batch_sizes.name) <= allowed_max_length
  assert len(result.rf_flags.name) <= allowed_max_length
