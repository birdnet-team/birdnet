import multiprocessing as mp

import numpy as np

from birdnet.acoustic.inference.core.prediction.prediction_tensor import (
  AcousticPredictionTensor,
)


def test_resize_keeps_unwritten_prediction_segments_masked() -> None:
  max_segment_index = mp.RawValue("I", 11)
  tensor = AcousticPredictionTensor(
    session_id="test",
    n_inputs=4,
    top_k=4,
    n_species=10,
    half_precision=False,
    input_indices_dtype=np.dtype(np.uint8),
    segment_indices_dtype=np.dtype(np.uint8),
    max_segment_index=max_segment_index,
  )

  top_k_species = np.arange(4, dtype=tensor._species_ids.dtype)[None, :]
  top_k_scores = np.ones((1, 4), dtype=tensor._species_probs.dtype)
  top_k_mask = np.zeros((1, 4), dtype=bool)

  for segment_index in range(4):
    tensor.write_block(
      np.array([3], dtype=np.uint8),
      np.array([segment_index], dtype=np.uint8),
      top_k_species,
      top_k_scores,
      top_k_mask,
    )

  for max_index in (14, 17, 20):
    max_segment_index.value = max_index
    tensor.write_block(
      np.array([0], dtype=np.uint8),
      np.array([max_index], dtype=np.uint8),
      top_k_species,
      top_k_scores,
      top_k_mask,
    )

  assert np.all(tensor._species_masked[3, 4:, :])
