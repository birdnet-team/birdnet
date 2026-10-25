"""An encoding result without a single row writes to Parquet."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticDataEncodingResult,
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet_tests.acoustic_models.inference.encoding.encoding_result_py.test_encoding_to_structured_array import (  # noqa: E501
  create_file_encoding_result,
  create_mock_encoding_tensor,
)

pytestmark = pytest.mark.no_tf

_ALL_SEGMENTS = [(f, s) for f in range(2) for s in range(2)]


def _expected_schema(input_values: pa.DataType) -> pa.Schema:
  return pa.schema(
    [
      pa.field("input", pa.dictionary(pa.int32(), input_values), nullable=False),
      pa.field("start_time", pa.float32(), nullable=False),
      pa.field("end_time", pa.float32(), nullable=False),
      pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ]
  )


def _file_result(*, masked: bool) -> AcousticFileEncodingResult:
  return create_file_encoding_result(
    n_files=2,
    duration_s=6,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    masked_segments=_ALL_SEGMENTS if masked else None,
  )


def _array_result(*, masked: bool) -> AcousticDataEncodingResult:
  return AcousticDataEncodingResult(
    tensor=create_mock_encoding_tensor(
      2, 2, 6, masked_segments=_ALL_SEGMENTS if masked else None
    ),
    input_durations=np.full(2, 6, dtype=np.float16),
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    speed=1.0,
    model_path=Path("/model/path"),
    model_fmin=0,
    model_fmax=15_000,
    model_sr=48_000,
    model_precision="fp32",
    model_version="v2.4",
  )


def _assert_same_schema_as_non_empty(
  empty: AcousticEncodingResultBase,
  reference: AcousticEncodingResultBase,
  expected: pa.Schema,
  tmp_path: Path,
) -> None:
  empty.to_parquet(tmp_path / "empty.parquet", silent=True)
  reference.to_parquet(tmp_path / "reference.parquet", silent=True)

  reference_table = reference.to_arrow_table()
  assert reference_table.num_rows > 0
  assert reference_table.schema.equals(expected)
  assert empty.to_arrow_table().schema.equals(
    reference_table.schema, check_metadata=True
  )
  written = pq.read_table(tmp_path / "empty.parquet")
  assert written.num_rows == 0
  assert written.schema.equals(
    pq.read_table(tmp_path / "reference.parquet").schema, check_metadata=True
  )


def test_file_result_without_rows_has_the_non_empty_schema(tmp_path: Path) -> None:
  _assert_same_schema_as_non_empty(
    _file_result(masked=True),
    _file_result(masked=False),
    _expected_schema(pa.string()),
    tmp_path,
  )


def test_array_result_without_rows_has_the_non_empty_schema(tmp_path: Path) -> None:
  """Integer array inputs are not forced to strings; they infer a type even empty."""
  _assert_same_schema_as_non_empty(
    _array_result(masked=True),
    _array_result(masked=False),
    _expected_schema(pa.uint8()),
    tmp_path,
  )
