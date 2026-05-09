from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet_tests.acoustic_models.inference.encoding.encoding_result_py.test_encoding_to_structured_array import (  # noqa: E501
  create_file_encoding_result,
)


def _create_result_with_float16_durations() -> AcousticFileEncodingResult:
  """Create an encoding result whose input_durations are float16."""
  result = create_file_encoding_result(
    n_files=2,
    duration_s=12,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )
  assert result.input_durations.dtype == np.float16
  return result


def _create_result_with_float32_durations() -> AcousticFileEncodingResult:
  """Create an encoding result whose input_durations are float32."""
  result = create_file_encoding_result(
    n_files=1,
    duration_s=5000,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )
  assert result.input_durations.dtype == np.float32
  return result


def _create_result_with_float64_durations() -> AcousticFileEncodingResult:
  """Create an encoding result whose input_durations are float64.

  Uses a small duration for speed, then coerces dtype to float64 to exercise
  the Arrow type-promotion path without creating millions of segments.
  """
  result = create_file_encoding_result(
    n_files=1,
    duration_s=12,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )
  result._input_durations = result._input_durations.astype(np.float64)
  assert result.input_durations.dtype == np.float64
  return result


def test_arrow_table_time_columns_are_float32_when_durations_float16() -> None:
  result = _create_result_with_float16_durations()
  table = result.to_arrow_table()

  assert table.schema.field("start_time").type == pa.float32()
  assert table.schema.field("end_time").type == pa.float32()


def test_arrow_table_time_columns_are_float32_when_durations_float32() -> None:
  result = _create_result_with_float32_durations()
  table = result.to_arrow_table()

  assert table.schema.field("start_time").type == pa.float32()
  assert table.schema.field("end_time").type == pa.float32()


def test_arrow_table_time_columns_are_float64_when_durations_float64() -> None:
  result = _create_result_with_float64_durations()
  table = result.to_arrow_table()

  assert table.schema.field("start_time").type == pa.float64()
  assert table.schema.field("end_time").type == pa.float64()


def test_parquet_roundtrip_schema_float16(tmp_path: Path) -> None:
  result = _create_result_with_float16_durations()
  out = tmp_path / "result.parquet"

  result.to_parquet(out, silent=True)
  table = pq.read_table(out)

  assert table.schema.field("start_time").type == pa.float32()
  assert table.schema.field("end_time").type == pa.float32()


def test_parquet_roundtrip_values_float16(tmp_path: Path) -> None:
  result = _create_result_with_float16_durations()
  structured = result.to_structured_array()
  out = tmp_path / "result.parquet"

  expected_start = np.array(structured["start_time"], dtype=np.float64)
  expected_end = np.array(structured["end_time"], dtype=np.float64)

  result.to_parquet(out, silent=True)
  table = pq.read_table(out)

  actual_start = np.array(table.column("start_time").to_pylist(), dtype=np.float64)
  actual_end = np.array(table.column("end_time").to_pylist(), dtype=np.float64)

  np.testing.assert_allclose(expected_start, actual_start, rtol=1e-3)
  np.testing.assert_allclose(expected_end, actual_end, rtol=1e-3)


def test_parquet_roundtrip_values_float32(tmp_path: Path) -> None:
  result = _create_result_with_float32_durations()
  structured = result.to_structured_array()
  out = tmp_path / "result.parquet"

  expected_start = np.array(structured["start_time"], dtype=np.float64)
  expected_end = np.array(structured["end_time"], dtype=np.float64)

  result.to_parquet(out, silent=True)
  table = pq.read_table(out)

  actual_start = np.array(table.column("start_time").to_pylist(), dtype=np.float64)
  actual_end = np.array(table.column("end_time").to_pylist(), dtype=np.float64)

  np.testing.assert_allclose(expected_start, actual_start, rtol=1e-6)
  np.testing.assert_allclose(expected_end, actual_end, rtol=1e-6)


def test_parquet_roundtrip_values_float64(tmp_path: Path) -> None:
  result = _create_result_with_float64_durations()
  structured = result.to_structured_array()
  out = tmp_path / "result.parquet"

  expected_start = np.array(structured["start_time"], dtype=np.float64)
  expected_end = np.array(structured["end_time"], dtype=np.float64)

  result.to_parquet(out, silent=True)
  table = pq.read_table(out)

  actual_start = np.array(table.column("start_time").to_pylist(), dtype=np.float64)
  actual_end = np.array(table.column("end_time").to_pylist(), dtype=np.float64)

  np.testing.assert_allclose(expected_start, actual_start, rtol=1e-9)
  np.testing.assert_allclose(expected_end, actual_end, rtol=1e-9)


def test_parquet_time_columns_no_halffloat(tmp_path: Path) -> None:
  """Ensure start_time and end_time never use halffloat in Parquet."""
  result = _create_result_with_float16_durations()
  out = tmp_path / "result.parquet"

  result.to_parquet(out, silent=True)
  table = pq.read_table(out)

  for col_name in ("start_time", "end_time"):
    field = table.schema.field(col_name)
    assert field.type != pa.float16(), (
      f"Column '{col_name}' uses halffloat (float16), "
      f"which is not interoperable across Arrow implementations"
    )
