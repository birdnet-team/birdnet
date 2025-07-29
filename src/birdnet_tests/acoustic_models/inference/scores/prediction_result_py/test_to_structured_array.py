from pathlib import Path

import numpy as np
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.scores.prediction_result import (
  PredictionResult,
  assert_species_masked_pattern,
)
from birdnet.acoustic_models.inference.scores.tensor import ScoresTensor
from birdnet.helper import get_float_dtype


def create_mock_tensor(
  species_ids: np.ndarray, species_probs: np.ndarray, species_masked: np.ndarray
) -> ScoresTensor:
  """Helper to create a mock ScoresTensor."""
  tensor = ScoresTensor.__new__(ScoresTensor)
  tensor._species_ids = species_ids
  tensor._species_probs = species_probs
  tensor._species_masked = species_masked
  return tensor


def create_prediction_result(
  n_files: int,
  n_segments: int,
  top_k: int,
  segment_duration_s: float,
  overlap_duration_s: float,
  rand_duration: bool = False,
) -> PredictionResult:
  np.random.seed(0)
  species_ids = np.random.randint(
    0, 10, size=(n_files, n_segments, top_k), dtype=np.uint8
  )
  species_probs = np.random.random((n_files, n_segments, top_k)).astype(np.float32)
  species_masked = np.full((n_files, n_segments, top_k), False, dtype=bool)

  tensor = create_mock_tensor(species_ids, species_probs, species_masked)

  files = OrderedSet([Path(f"/test/file_{i}.wav") for i in range(n_files)])
  species_list = OrderedSet([f"species_{i}" for i in range(15)])
  if rand_duration:
    # rand value [0.5, 2.5] * n_segments
    file_durations = np.random.choice(
      np.arange(((n_segments - 1) * 3) + 0.5, (n_segments * 3), 0.5), size=n_files
    )
    file_durations = file_durations.astype(get_float_dtype(max(file_durations)))
  else:
    max_dur = n_segments * segment_duration_s
    file_durations = np.full(
      n_files,
      max_dur,
      dtype=get_float_dtype(max_dur),
    )
  return PredictionResult(
    tensor=tensor,
    files=files,
    species_list=species_list,
    file_durations=file_durations,
    segment_duration_s=segment_duration_s,
    overlap_duration_s=overlap_duration_s,
  )


def test_empty_predictions() -> None:
  result = create_prediction_result(
    n_files=5, n_segments=3, top_k=2, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  result.species_masked[:] = True

  structured = result.to_structured_array()

  assert len(structured) == 0
  assert structured.dtype.names == (
    "file_path",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  )


def test_single_prediction() -> None:
  result = create_prediction_result(
    n_files=1, n_segments=1, top_k=1, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  assert len(structured) == 1
  assert structured[0]["file_path"] == "/test/file_0.wav"
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert str(structured[0]["species_name"]).startswith("species_")
  assert structured[0]["confidence"] >= 0


def test_two_segments() -> None:
  result = create_prediction_result(
    n_files=1, n_segments=2, top_k=1, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["file_path"] == "/test/file_0.wav"
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert str(structured[0]["species_name"]).startswith("species_")
  assert structured[0]["confidence"] >= 0

  assert structured[1]["file_path"] == "/test/file_0.wav"
  assert structured[1]["start_time"] == 3.0
  assert structured[1]["end_time"] == 6.0
  assert str(structured[1]["species_name"]).startswith("species_")
  assert structured[1]["confidence"] >= 0


def test_sorting_by_confidence() -> None:
  result = create_prediction_result(
    n_files=1, n_segments=1, top_k=3, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  assert len(structured) == 3
  assert (
    structured[0]["confidence"]
    >= structured[1]["confidence"]
    >= structured[2]["confidence"]
  )


def test_time_calculations_with_overlap() -> None:
  result = create_prediction_result(
    n_files=1, n_segments=2, top_k=1, segment_duration_s=3, overlap_duration_s=0.5
  )

  structured = result.to_structured_array()

  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert structured[1]["start_time"] == 2.5
  assert structured[1]["end_time"] == 5.5


def test_end_time_clipping_one_segment() -> None:
  result = create_prediction_result(
    n_files=1,
    n_segments=1,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    rand_duration=True,
  )

  structured = result.to_structured_array()

  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] < 3.0


def test_end_time_clipping_two_segments() -> None:
  result = create_prediction_result(
    n_files=1,
    n_segments=2,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    rand_duration=True,
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert structured[1]["start_time"] == 3.0
  assert structured[1]["end_time"] < 6.0


def _test_end_time_clipping_multiple_segments(
  max_duration: float,
) -> PredictionResult:
  n_segments = round(max_duration / 3)
  result = create_prediction_result(
    n_files=1,
    n_segments=n_segments,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    rand_duration=True,
  )

  structured = result.to_structured_array()

  assert len(structured) == n_segments
  for i in range(n_segments):
    assert structured[i]["start_time"] == 3 * i, (
      f"Start time mismatch at index {i}, expected {3 * i}, got {structured[i]['start_time']}"
    )
  for i in range(n_segments - 1):
    assert structured[i]["end_time"] == 3 * (i + 1)
  assert structured[-1]["end_time"] < 3 * n_segments
  return result


def test_end_time_clipping_multiple_segments_float16() -> None:
  result = _test_end_time_clipping_multiple_segments(2000)
  assert result.file_durations.dtype == np.float16


def test_end_time_clipping_multiple_segments_float32() -> None:
  result = _test_end_time_clipping_multiple_segments(5000)
  assert result.file_durations.dtype == np.float32


def xtest_end_time_clipping_multiple_segments_float64() -> None:
  # takes to long to test
  result = _test_end_time_clipping_multiple_segments(2**25)
  assert result.file_durations.dtype == np.float64


def test_multiple_files() -> None:
  result = create_prediction_result(
    n_files=5, n_segments=1, top_k=1, segment_duration_s=3, overlap_duration_s=0
  )

  structured = result.to_structured_array()

  assert len(structured) == 5
  assert structured[0]["file_path"] == "/test/file_0.wav"
  assert structured[1]["file_path"] == "/test/file_1.wav"
  assert structured[2]["file_path"] == "/test/file_2.wav"
  assert structured[3]["file_path"] == "/test/file_3.wav"
  assert structured[4]["file_path"] == "/test/file_4.wav"


def test_dtype_structure() -> None:
  result = create_prediction_result(
    n_files=1, n_segments=1, top_k=1, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  expected_fields = [
    "file_path",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  ]
  assert structured.dtype.names == tuple(expected_fields)
  assert structured.dtype["file_path"] == np.dtype("O")
  assert structured.dtype["start_time"] == result._file_durations.dtype
  assert structured.dtype["end_time"] == result._file_durations.dtype
  assert structured.dtype["species_name"] == np.dtype("O")
  assert structured.dtype["confidence"] == result._species_probs.dtype


def test_masking_behavior() -> None:
  result = create_prediction_result(
    n_files=2, n_segments=1, top_k=3, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  result.species_masked[:] = True
  result.species_masked[0, 0, 0] = False
  result.species_masked[0, 0, 1] = False
  result.species_masked[1, 0, 0] = False
  assert_species_masked_pattern(result.species_masked)

  structured = result.to_structured_array()

  assert len(structured) == 3
  assert structured[0]["file_path"] == "/test/file_0.wav"
  assert structured[1]["file_path"] == "/test/file_0.wav"
  assert structured[2]["file_path"] == "/test/file_1.wav"
