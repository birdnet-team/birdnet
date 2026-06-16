import multiprocessing as mp

import numpy as np

from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
  AcousticEncodingTensor,
)


def test_resize_keeps_unwritten_encoding_segments_masked() -> None:
  max_segment_index = mp.RawValue("I", 11)
  tensor = AcousticEncodingTensor(
    session_id="test",
    n_inputs=4,
    emb_dim=2,
    half_precision=False,
    input_indices_dtype=np.dtype(np.uint8),
    segment_indices_dtype=np.dtype(np.uint8),
    max_segment_index=max_segment_index,
  )

  emb = np.ones((1, 2), dtype=tensor._emb.dtype)

  for segment_index in range(4):
    tensor.write_block(
      np.array([3], dtype=np.uint8),
      np.array([segment_index], dtype=np.uint8),
      emb,
    )

  for max_index in (14, 17, 20):
    max_segment_index.value = max_index
    tensor.write_block(
      np.array([0], dtype=np.uint8),
      np.array([max_index], dtype=np.uint8),
      emb,
    )

  assert np.all(tensor._emb_masked[3, 4:, :])