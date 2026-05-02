from pathlib import Path

import numpy as np

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
  AcousticEncodingTensor,
)
from birdnet.utils.helper import (
  get_float_dtype,
  get_hop_duration_s,
  get_n_segments_speed,
)

DEFAULT_EMBEDDING_DIM = 6


def create_mock_encoding_tensor(
  n_files: int,
  n_segments: int,
  embedding_dim: int,
  dtype: np.dtype = np.float32,
  masked_segments: list[tuple[int, int]] | None = None,
  unprocessable_files: set[int] | None = None,
) -> AcousticEncodingTensor:
  total_values = n_files * n_segments * embedding_dim
  raw_values = np.arange(total_values, dtype=np.float32)
  embeddings = raw_values.reshape(n_files, n_segments, embedding_dim).astype(dtype)
  mask = np.zeros_like(embeddings, dtype=bool)
  if masked_segments:
    for file_idx, segment_idx in masked_segments:
      mask[file_idx, segment_idx, :] = True

  tensor = AcousticEncodingTensor.__new__(AcousticEncodingTensor)
  tensor._emb = embeddings
  tensor._emb_masked = mask
  tensor.set_unprocessable_inputs(unprocessable_files or set())
  return tensor


def create_file_encoding_result(
  n_files: int,
  duration_s: float,
  segment_duration_s: float,
  overlap_duration_s: float,
  speed: float = 1.0,
  embedding_dim: int = DEFAULT_EMBEDDING_DIM,
  masked_segments: list[tuple[int, int]] | None = None,
  unprocessable_files: set[int] | None = None,
) -> AcousticFileEncodingResult:
  assert n_files > 0
  assert segment_duration_s > overlap_duration_s
  n_segments = get_n_segments_speed(
    duration_s, segment_duration_s, overlap_duration_s, speed
  )
  tensor = create_mock_encoding_tensor(
    n_files,
    n_segments,
    embedding_dim,
    masked_segments=masked_segments,
    unprocessable_files=unprocessable_files,
  )
  files = [Path(f"/test/file_{i}.wav") for i in range(n_files)]
  file_durations = np.full(
    n_files,
    duration_s,
    dtype=get_float_dtype(duration_s),
  )
  return AcousticFileEncodingResult(
    tensor=tensor,
    files=files,
    file_durations=file_durations,
    segment_duration_s=segment_duration_s,
    overlap_duration_s=overlap_duration_s,
    speed=speed,
    model_path=Path("/model/path"),
    model_fmin=0,
    model_fmax=15_000,
    model_sr=48_000,
    model_precision="fp32",
    model_version="v2.4",
  )


def test_empty_embeddings_returns_empty() -> None:
  segment_duration = 3.0
  overlap_duration = 0.0
  n_files = 2
  duration = 6.0
  n_segments = get_n_segments_speed(duration, segment_duration, overlap_duration, 1.0)
  masked_segments = [(f, s) for f in range(n_files) for s in range(n_segments)]
  result = create_file_encoding_result(
    n_files=n_files,
    duration_s=duration,
    segment_duration_s=segment_duration,
    overlap_duration_s=overlap_duration,
    masked_segments=masked_segments,
  )

  structured = result.to_structured_array()

  assert len(structured) == 0
  assert structured.dtype.names == (
    "input",
    "start_time",
    "end_time",
    "embedding",
  )


def test_embedding_vector_matches_tensor() -> None:
  result = create_file_encoding_result(
    n_files=1,
    duration_s=6,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )

  structured = result.to_structured_array()

  np.testing.assert_array_equal(
    structured[0]["embedding"], np.arange(DEFAULT_EMBEDDING_DIM, dtype=np.float32)
  )
  np.testing.assert_array_equal(
    structured[1]["embedding"],
    np.arange(DEFAULT_EMBEDDING_DIM, dtype=np.float32) + DEFAULT_EMBEDDING_DIM,
  )


def test_unprocessable_inputs_are_removed() -> None:
  result = create_file_encoding_result(
    n_files=1,
    duration_s=3,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    unprocessable_files={0},
  )

  structured = result.to_structured_array()

  assert len(structured) == 0


def test_masked_segments_are_skipped() -> None:
  n_segments = get_n_segments_speed(6, 3, 0, 1.0)
  result = create_file_encoding_result(
    n_files=2,
    duration_s=6,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    masked_segments=[(0, 1), (1, 0)],
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[1]["input"] == str(Path("/test/file_1.wav").absolute())
  assert structured[0]["start_time"] == 0.0
  assert structured[1]["start_time"] == 3.0
  np.testing.assert_allclose(structured[0]["embedding"][0], 0.0)
  np.testing.assert_allclose(
    structured[1]["embedding"][0], DEFAULT_EMBEDDING_DIM * ((1 * n_segments) + 1)
  )


def test_time_calculations_no_overlap() -> None:
  duration = 6.0
  segment_duration = 3.0
  overlap_duration = 0.0
  speed = 1.0
  result = create_file_encoding_result(
    n_files=1,
    duration_s=duration,
    segment_duration_s=segment_duration,
    overlap_duration_s=overlap_duration,
    speed=speed,
  )

  structured = result.to_structured_array()
  hop = get_hop_duration_s(segment_duration, overlap_duration, speed)
  expected_starts = np.arange(len(structured)) * hop
  expected_ends = np.minimum(
    expected_starts + segment_duration * speed, result.input_durations[0]
  )

  np.testing.assert_allclose(structured["start_time"], expected_starts)
  np.testing.assert_allclose(structured["end_time"], expected_ends)


def test_time_calculations_with_overlap() -> None:
  duration = 6.0
  overlap_duration = 0.5
  segment_duration = 3.0
  speed = 1.0
  result = create_file_encoding_result(
    n_files=1,
    duration_s=duration,
    segment_duration_s=segment_duration,
    overlap_duration_s=overlap_duration,
    speed=speed,
  )

  structured = result.to_structured_array()
  hop = get_hop_duration_s(segment_duration, overlap_duration, speed)
  expected_starts = np.arange(len(structured)) * hop
  expected_ends = np.minimum(
    expected_starts + segment_duration * speed, result.input_durations[0]
  )

  np.testing.assert_allclose(structured["start_time"], expected_starts)
  np.testing.assert_allclose(structured["end_time"], expected_ends)


def test_time_calculations_speedup_halftime_no_overlap() -> None:
  duration = 6.0
  result = create_file_encoding_result(
    n_files=1,
    duration_s=duration,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    speed=0.5,
  )

  structured = result.to_structured_array()
  hop = get_hop_duration_s(3.0, 0.0, 0.5)
  expected_starts = np.arange(len(structured)) * hop
  expected_ends = np.minimum(expected_starts + 3.0 * 0.5, result.input_durations[0])

  np.testing.assert_allclose(structured["start_time"], expected_starts)
  np.testing.assert_allclose(structured["end_time"], expected_ends)


def test_time_calculations_speedup_doubletime_no_overlap() -> None:
  duration = 24.0
  result = create_file_encoding_result(
    n_files=1,
    duration_s=duration,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    speed=2.0,
  )

  structured = result.to_structured_array()
  hop = get_hop_duration_s(3.0, 0.0, 2.0)
  expected_starts = np.arange(len(structured)) * hop
  expected_ends = np.minimum(expected_starts + 3.0 * 2.0, result.input_durations[0])

  np.testing.assert_allclose(structured["start_time"], expected_starts)
  np.testing.assert_allclose(structured["end_time"], expected_ends)


def test_time_calculations_speedup_one_tenth_no_overlap() -> None:
  duration = 6.0
  result = create_file_encoding_result(
    n_files=1,
    duration_s=duration,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    speed=0.1,
  )

  structured = result.to_structured_array()
  hop = get_hop_duration_s(3.0, 0.0, 0.1)
  expected_starts = np.arange(len(structured)) * hop
  expected_ends = np.minimum(expected_starts + 3.0 * 0.1, result.input_durations[0])

  np.testing.assert_allclose(structured["start_time"], expected_starts)
  np.testing.assert_allclose(structured["end_time"], expected_ends)


def test_time_calculations_speedup_decimal_no_overlap() -> None:
  duration = 6.0
  result = create_file_encoding_result(
    n_files=1,
    duration_s=duration,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    speed=0.1387434856,
  )

  structured = result.to_structured_array()
  hop = get_hop_duration_s(3.0, 0.0, 0.1387434856)
  expected_starts = np.arange(len(structured)) * hop
  expected_ends = np.minimum(
    expected_starts + 3.0 * 0.1387434856, result.input_durations[0]
  )

  np.testing.assert_allclose(structured["start_time"], expected_starts)
  np.testing.assert_allclose(structured["end_time"], expected_ends)


def _test_end_time_clipping_multiple_segments(
  max_duration: float,
) -> AcousticEncodingResultBase:
  result = create_file_encoding_result(
    n_files=1,
    duration_s=max_duration,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )

  structured = result.to_structured_array()
  n_segments = get_n_segments_speed(max_duration, 3.0, 0.0, 1.0)

  assert len(structured) == n_segments
  for idx in range(n_segments):
    np.testing.assert_allclose(structured[idx]["start_time"], idx * 3.0)
    if idx < n_segments - 1:
      expected_end = (idx + 1) * 3.0
    else:
      expected_end = result.input_durations[0]
    np.testing.assert_allclose(structured[idx]["end_time"], expected_end)

  return result


def test_end_time_clipping_multiple_segments_float16() -> None:
  result = _test_end_time_clipping_multiple_segments(2_000.0)
  assert result.input_durations.dtype == np.float16


def test_end_time_clipping_multiple_segments_float32() -> None:
  result = _test_end_time_clipping_multiple_segments(5_000.0)
  assert result.input_durations.dtype == np.float32


def xtest_end_time_clipping_multiple_segments_float64() -> None:
  result = _test_end_time_clipping_multiple_segments(2**25)
  assert result.input_durations.dtype == np.float64


def test_multiple_files() -> None:
  result = create_file_encoding_result(
    n_files=5,
    duration_s=3,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )

  structured = result.to_structured_array()

  assert len(structured) == 5
  for idx in range(5):
    assert structured[idx]["input"] == str(Path(f"/test/file_{idx}.wav").absolute())


def test_dtype_structure() -> None:
  result = create_file_encoding_result(
    n_files=1,
    duration_s=3,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )

  structured = result.to_structured_array()
  embedding_dtype = structured.dtype["embedding"]

  assert structured.dtype.names == (
    "input",
    "start_time",
    "end_time",
    "embedding",
  )
  assert structured.dtype["input"] == np.dtype("O")
  assert structured.dtype["start_time"] == result._input_durations.dtype
  assert structured.dtype["end_time"] == result._input_durations.dtype
  assert embedding_dtype.shape == (DEFAULT_EMBEDDING_DIM,)
  assert embedding_dtype.base == np.dtype(np.float32)
