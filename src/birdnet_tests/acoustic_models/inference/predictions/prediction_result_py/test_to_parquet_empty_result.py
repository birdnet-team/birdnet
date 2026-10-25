"""A prediction result without a single row writes to Parquet."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
  AcousticPredictionResultBase,
)
from birdnet_tests.acoustic_models.inference.predictions.prediction_result_py.test_prediction_to_structured_array import (  # noqa: E501
  create_file_prediction_result,
  create_mock_scores_tensor,
)

pytestmark = pytest.mark.no_tf


def _expected_schema(input_values: pa.DataType) -> pa.Schema:
  return pa.schema(
    [
      pa.field("input", pa.dictionary(pa.int32(), input_values), nullable=False),
      pa.field("start_time", pa.float32(), nullable=False),
      pa.field("end_time", pa.float32(), nullable=False),
      pa.field("species_name", pa.dictionary(pa.int32(), pa.string()), nullable=False),
      pa.field("confidence", pa.float32(), nullable=False),
    ]
  )


def _file_result(*, masked: bool) -> AcousticFilePredictionResult:
  result = create_file_prediction_result(
    n_files=2, duration_s=9, top_k=2, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  if masked:
    result.species_masked[:] = True
  return result


def _array_result(*, masked: bool) -> AcousticDataPredictionResult:
  shape = (2, 3, 2)
  tensor = create_mock_scores_tensor(
    np.zeros(shape, dtype=np.uint8),
    np.full(shape, 0.5, dtype=np.float32),
    np.full(shape, masked),
  )
  return AcousticDataPredictionResult(
    tensor=tensor,
    species_list=OrderedSet(["species_0", "species_1"]),
    input_durations=np.full(2, 9, dtype=np.float16),
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
  empty: AcousticPredictionResultBase,
  reference: AcousticPredictionResultBase,
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
  """Array inputs stay integers; species names are an object column here too."""
  _assert_same_schema_as_non_empty(
    _array_result(masked=True),
    _array_result(masked=False),
    _expected_schema(pa.uint8()),
    tmp_path,
  )
