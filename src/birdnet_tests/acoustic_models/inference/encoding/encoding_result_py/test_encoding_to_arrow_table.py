"""The Arrow embedding column holds every embedding, row for row."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest

import birdnet.acoustic.inference.core.encoding.encoding_result as encoding_result
from birdnet.acoustic.inference.core.encoding.encoding_result import VAR_EMBEDDING
from birdnet_tests.acoustic_models.inference.encoding.encoding_result_py.test_encoding_to_structured_array import (  # noqa: E501
  create_file_encoding_result,
)

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_embedding_column_matches_the_structured_array(dtype: type) -> None:
  result = create_file_encoding_result(
    n_files=2,
    duration_s=12,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    masked_segments=[(0, 1)],
  )
  rng = np.random.default_rng(0)
  result._embeddings = rng.standard_normal(result._embeddings.shape).astype(dtype)

  column = result.to_arrow_table().column(VAR_EMBEDDING)

  assert column.type == pa.list_(pa.from_numpy_dtype(dtype))
  # The table schema casts, so check the column as built too.
  built = encoding_result._embedding_column(result.to_structured_array()[VAR_EMBEDDING])
  assert built.type == pa.list_(pa.from_numpy_dtype(dtype))
  assert column.type.value_field.name == "item"
  expected = result.to_structured_array()[VAR_EMBEDDING]
  np.testing.assert_array_equal(np.array(column.to_pylist(), dtype=dtype), expected)


def test_empty_result_has_an_empty_embedding_column() -> None:
  result = create_file_encoding_result(
    n_files=1,
    duration_s=6,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    masked_segments=[(0, 0), (0, 1)],
  )

  column = result.to_arrow_table().column(VAR_EMBEDDING)

  assert len(column) == 0
  assert column.type == pa.list_(pa.float32())


def test_a_column_beyond_the_int32_offsets_is_chunked(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  result = create_file_encoding_result(
    n_files=2, duration_s=12, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  expected = result.to_structured_array()[VAR_EMBEDDING]
  n_rows, emb_dim = expected.shape
  rows_per_chunk = 3
  assert n_rows > rows_per_chunk
  monkeypatch.setattr(
    encoding_result, "_MAX_VALUES_PER_CHUNK", rows_per_chunk * emb_dim
  )

  column = result.to_arrow_table().column(VAR_EMBEDDING)

  assert column.num_chunks == math.ceil(n_rows / rows_per_chunk)
  assert column.type == pa.list_(pa.float32())
  np.testing.assert_array_equal(
    np.array(column.to_pylist(), dtype=np.float32), expected
  )
