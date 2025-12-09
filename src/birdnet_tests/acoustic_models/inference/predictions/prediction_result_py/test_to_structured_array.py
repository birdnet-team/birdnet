from pathlib import Path

import numpy as np
import soundfile
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.prediction.result import (
  AcousticFilePredictionResult,
  AcousticPredictionResultBase,
)
from birdnet.acoustic_models.inference.prediction.tensor import AcousticPredictionTensor
from birdnet.helper import (
  get_float_dtype,
  get_n_segments_speed,
)
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG


def assert_species_masked_pattern(species_masked: np.ndarray) -> None:
  """
  Assert that species_masked has False entries first, then only True entries.
  For 3D arrays (n_files, n_segments, top_k), checks the pattern along the top_k axis
  for each (file, segment) combination.
  """
  if species_masked.size == 0:
    return

  assert species_masked.ndim == 3

  # Reshape to (n_files * n_segments, top_k) for vectorized processing
  n_files, n_segments, top_k = species_masked.shape
  reshaped = species_masked.reshape(-1, top_k)

  # For each row (file, segment combination), find first True
  # Using argmax on the mask gives us the first True position
  # If no True exists, argmax returns 0, but we handle this separately
  has_true = np.any(reshaped, axis=1)
  first_true_pos = np.argmax(reshaped, axis=1)

  # Only check rows that have at least one True
  if np.any(has_true):
    valid_rows = np.where(has_true)[0]

    for row_idx in valid_rows:
      row = reshaped[row_idx]
      first_true = first_true_pos[row_idx]

      # Quick check: all before first_true should be False, all after should be True
      if not (np.all(~row[:first_true]) and np.all(row[first_true:])):
        file_idx, seg_idx = divmod(row_idx, n_segments)
        raise AssertionError(
          f"Invalid mask pattern at file {file_idx}, segment {seg_idx}: "
          f"expected False...False,True...True pattern"
        )


def create_mock_scores_tensor(
  species_ids: np.ndarray,
  species_probs: np.ndarray,
  species_masked: np.ndarray,
  unprocessable_files: set[int] | None = None,
) -> AcousticPredictionTensor:
  """Helper to create a mock tensor."""
  tensor = AcousticPredictionTensor.__new__(AcousticPredictionTensor)
  tensor._species_ids = species_ids
  tensor._species_probs = species_probs
  tensor._species_masked = species_masked
  tensor.set_unprocessable_inputs(unprocessable_files or set())
  return tensor


def create_file_prediction_result(
  n_files: int,
  duration_s: float,
  top_k: int,
  segment_duration_s: float,
  overlap_duration_s: float,
  speed: float = 1.0,
  unprocessable_files: set[int] | None = None,
) -> AcousticFilePredictionResult:
  assert n_files > 0
  assert 0 <= overlap_duration_s < segment_duration_s
  np.random.seed(0)
  n_segments = get_n_segments_speed(
    duration_s, segment_duration_s, overlap_duration_s, speed
  )
  species_ids = np.random.randint(
    0, 10, size=(n_files, n_segments, top_k), dtype=np.uint8
  )
  species_probs = np.random.random((n_files, n_segments, top_k)).astype(np.float32)
  species_masked = np.full((n_files, n_segments, top_k), False, dtype=bool)

  tensor = create_mock_scores_tensor(
    species_ids, species_probs, species_masked, unprocessable_files
  )

  files = [Path(f"/test/file_{i}.wav") for i in range(n_files)]
  species_list = OrderedSet([f"species_{i}" for i in range(15)])
  file_durations = np.full(
    n_files,
    duration_s,
    dtype=get_float_dtype(duration_s),
  )
  return AcousticFilePredictionResult(
    tensor=tensor,
    files=files,
    species_list=species_list,
    file_durations=file_durations,
    segment_duration_s=segment_duration_s,
    overlap_duration_s=overlap_duration_s,
    speed=speed,
    model_fmax=15_000,
    model_fmin=0,
    model_path=Path("/model/path"),
    model_precision="fp32",
    model_sr=48_000,
    model_version="v2.4",
  )


def test_empty_predictions() -> None:
  result = create_file_prediction_result(
    n_files=5, duration_s=9, top_k=2, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  result.species_masked[:] = True

  structured = result.to_structured_array()

  assert len(structured) == 0
  assert structured.dtype.names == (
    "input",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  )


def test_single_prediction() -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=3, top_k=1, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  assert len(structured) == 1
  assert structured[0]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert str(structured[0]["species_name"]).startswith("species_")
  assert structured[0]["confidence"] >= 0


def test_unprocessable_returns_empty() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=3,
    top_k=1,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    unprocessable_files={0},
  )

  structured = result.to_structured_array()

  assert len(structured) == 0


def test_one_unprocessable_among_multiple_files() -> None:
  result = create_file_prediction_result(
    n_files=3,
    duration_s=3,
    top_k=1,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    unprocessable_files={1},
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[1]["input"] == str(Path("/test/file_2.wav").absolute())


def test_two_segments() -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=6, top_k=1, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert str(structured[0]["species_name"]).startswith("species_")
  assert structured[0]["confidence"] >= 0

  assert structured[1]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[1]["start_time"] == 3.0
  assert structured[1]["end_time"] == 6.0
  assert str(structured[1]["species_name"]).startswith("species_")
  assert structured[1]["confidence"] >= 0


def test_sorting_by_confidence() -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=3, top_k=3, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  assert len(structured) == 3
  assert (
    structured[0]["confidence"]
    >= structured[1]["confidence"]
    >= structured[2]["confidence"]
  )


def test_time_calculations_no_overlap() -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=6, top_k=1, segment_duration_s=3, overlap_duration_s=0
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert structured[1]["start_time"] == 3.0
  assert structured[1]["end_time"] == 6.0
  assert structured[1]["end_time"] == result.input_durations[0]


def test_time_calculations_with_overlap() -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=6, top_k=1, segment_duration_s=3, overlap_duration_s=0.5
  )

  structured = result.to_structured_array()

  assert len(structured) == 3
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert structured[1]["start_time"] == 2.5
  assert structured[1]["end_time"] == 5.5
  assert structured[2]["start_time"] == 5.0
  assert structured[2]["end_time"] == 6.0
  assert structured[2]["end_time"] == result.input_durations[0]


def test_time_calculations_speedup_halftime_no_overlap() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=6,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    speed=0.5,
  )

  structured = result.to_structured_array()

  assert len(structured) == 4
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 1.5
  assert structured[1]["start_time"] == 1.5
  assert structured[1]["end_time"] == 3.0
  assert structured[2]["start_time"] == 3.0
  assert structured[2]["end_time"] == 4.5
  assert structured[3]["start_time"] == 4.5
  assert structured[3]["end_time"] == 6.0
  assert structured[3]["end_time"] == result.input_durations[0]


def test_random_time_calculations_speedup_halftime_no_overlap() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=5.75,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    speed=0.5,
  )

  structured = result.to_structured_array()

  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 1.5
  assert structured[1]["start_time"] == 1.5
  assert structured[1]["end_time"] == 3.0
  assert structured[2]["start_time"] == 3.0
  assert structured[2]["end_time"] == 4.5
  assert structured[3]["start_time"] == 4.5
  assert structured[3]["end_time"] == 5.75
  assert structured[3]["end_time"] == result.input_durations[0]
  assert len(structured) == 4


def test_time_calculations_speedup_doubletime_no_overlap() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=24,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    speed=2.0,
  )

  structured = result.to_structured_array()

  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 6.0
  assert structured[1]["start_time"] == 6.0
  assert structured[1]["end_time"] == 12.0
  assert structured[2]["start_time"] == 12.0
  assert structured[2]["end_time"] == 18.0
  assert structured[3]["start_time"] == 18.0
  assert structured[3]["end_time"] == 24.0
  assert structured[3]["end_time"] == result.input_durations[0]
  assert len(structured) == 4


def test_random_time_calculations_speedup_doubletime_no_overlap() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=20,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
    speed=2.0,
  )

  structured = result.to_structured_array()

  assert len(structured) == 4
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 6.0
  assert structured[1]["start_time"] == 6.0
  assert structured[1]["end_time"] == 12.0
  assert structured[2]["start_time"] == 12.0
  assert structured[2]["end_time"] == 18.0
  assert structured[3]["start_time"] == 18.0
  assert structured[3]["end_time"] == 20.0
  assert structured[3]["end_time"] == result.input_durations[0]


def test_time_calculations_with_overlap_and_speed() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=3,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0.5,
    speed=0.5,
  )

  structured = result.to_structured_array()

  assert len(structured) == 3
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 1.5
  assert structured[1]["start_time"] == 1.25  # (3-0.5) * 0.5
  assert structured[1]["end_time"] == 2.75
  assert structured[2]["start_time"] == 2.5
  assert structured[2]["end_time"] == 3.0
  assert structured[2]["end_time"] == result.input_durations[0]


def test_end_time_clipping_one_segment() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=2.75674,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
  )

  structured = result.to_structured_array()

  assert len(structured) == 1
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] < 3.0
  assert structured[0]["end_time"] == result.input_durations[0]


def test_end_time_clipping_two_segments() -> None:
  result = create_file_prediction_result(
    n_files=1,
    duration_s=5.7567,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
  )

  structured = result.to_structured_array()

  assert len(structured) == 2
  assert structured[0]["start_time"] == 0.0
  assert structured[0]["end_time"] == 3.0
  assert structured[1]["start_time"] == 3.0
  assert structured[1]["end_time"] < 6.0
  assert structured[1]["end_time"] == result.input_durations[0]


def _test_end_time_clipping_multiple_segments(
  max_duration: float,
) -> AcousticPredictionResultBase:
  n_segments = round(max_duration / 3)
  result = create_file_prediction_result(
    n_files=1,
    duration_s=max_duration,
    top_k=1,
    segment_duration_s=3,
    overlap_duration_s=0,
  )

  structured = result.to_structured_array()

  assert len(structured) == n_segments
  for i in range(n_segments):
    assert structured[i]["start_time"] == 3 * i, (
      f"Start time mismatch at index {i}, expected {3 * i}, "
      f"got {structured[i]['start_time']}"
    )
  for i in range(n_segments - 1):
    assert structured[i]["end_time"] == 3 * (i + 1)
  assert structured[-1]["end_time"] < 3 * n_segments
  return result


def test_end_time_clipping_multiple_segments_float16() -> None:
  result = _test_end_time_clipping_multiple_segments(2000)
  assert result.input_durations.dtype == np.float16


def test_end_time_clipping_multiple_segments_float32() -> None:
  result = _test_end_time_clipping_multiple_segments(5000)
  assert result.input_durations.dtype == np.float32


def xtest_end_time_clipping_multiple_segments_float64() -> None:
  # takes too long
  result = _test_end_time_clipping_multiple_segments(2**25)
  assert result.input_durations.dtype == np.float64


def test_multiple_files() -> None:
  result = create_file_prediction_result(
    n_files=5, duration_s=3, top_k=1, segment_duration_s=3, overlap_duration_s=0
  )

  structured = result.to_structured_array()

  assert len(structured) == 5
  assert structured[0]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[1]["input"] == str(Path("/test/file_1.wav").absolute())
  assert structured[2]["input"] == str(Path("/test/file_2.wav").absolute())
  assert structured[3]["input"] == str(Path("/test/file_3.wav").absolute())
  assert structured[4]["input"] == str(Path("/test/file_4.wav").absolute())


def test_dtype_structure() -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=3, top_k=1, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  structured = result.to_structured_array()

  expected_fields = [
    "input",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  ]
  assert structured.dtype.names == tuple(expected_fields)
  assert structured.dtype["input"] == np.dtype("O")
  assert structured.dtype["start_time"] == result._input_durations.dtype
  assert structured.dtype["end_time"] == result._input_durations.dtype
  assert structured.dtype["species_name"] == np.dtype("O")
  assert structured.dtype["confidence"] == result._species_probs.dtype


def test_masking_behavior() -> None:
  result = create_file_prediction_result(
    n_files=2, duration_s=3, top_k=3, segment_duration_s=3.0, overlap_duration_s=0.0
  )

  result.species_masked[:] = True
  result.species_masked[0, 0, 0] = False
  result.species_masked[0, 0, 1] = False
  result.species_masked[1, 0, 0] = False
  assert_species_masked_pattern(result.species_masked)

  structured = result.to_structured_array()

  assert len(structured) == 3
  assert structured[0]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[1]["input"] == str(Path("/test/file_0.wav").absolute())
  assert structured[2]["input"] == str(Path("/test/file_1.wav").absolute())


def test_full_pipeline() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1, top_k=1, speed=0.5, default_confidence_threshold=-np.inf
  ) as session:
    res = session.run(TEST_FILE_LONG)
  structured = res.to_structured_array()
  assert len(structured) == 80


def test_full_pipeline_np() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.predict_session(
    n_workers=1, top_k=1, speed=0.5, default_confidence_threshold=-np.inf
  ) as session:
    sf_read = soundfile.read(TEST_FILE_LONG)
    res = session.run_arrays(sf_read)
  structured = res.to_structured_array()
  assert len(structured) == 80
