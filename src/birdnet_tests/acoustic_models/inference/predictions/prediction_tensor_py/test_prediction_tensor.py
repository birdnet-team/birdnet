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


def test_initial_zero_pointer_keeps_prediction_tensor_empty_until_first_write() -> None:
  max_segment_index = mp.RawValue("I", 0)
  tensor = AcousticPredictionTensor(
    session_id="test",
    n_inputs=1,
    top_k=2,
    n_species=4,
    half_precision=False,
    input_indices_dtype=np.dtype(np.uint8),
    segment_indices_dtype=np.dtype(np.uint8),
    max_segment_index=max_segment_index,
  )

  assert tensor._species_probs.shape == (1, 0, 2)

  top_k_species = np.array([[0, 1]], dtype=tensor._species_ids.dtype)
  top_k_scores = np.array([[0.5, 0.25]], dtype=tensor._species_probs.dtype)
  top_k_mask = np.array([[False, False]], dtype=bool)

  tensor.write_block(
    np.array([0], dtype=np.uint8),
    np.array([0], dtype=np.uint8),
    top_k_species,
    top_k_scores,
    top_k_mask,
  )

  assert tensor._species_probs.shape == (1, 1, 2)
  assert np.all(~tensor._species_masked[0, 0])
