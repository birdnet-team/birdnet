from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
  AcousticEncodingTensor,
)
from birdnet.utils.dtypes import get_float_dtype
from birdnet.utils.segments import get_n_segments_speed
from birdnet_tests.helper import saved_arrays

pytestmark = pytest.mark.no_tf


def create_mock_tensor_one_unproc(
  emb: np.ndarray, emb_masked: np.ndarray
) -> AcousticEncodingTensor:
  tensor = AcousticEncodingTensor.__new__(AcousticEncodingTensor)
  tensor._emb = emb
  tensor._emb_masked = emb_masked
  tensor._unprocessable_inputs = np.array([0], dtype=np.uint8)
  return tensor


def create_dummy_result(
  n_files: int = 3,
  duration_s: float = 20.0,
  segment_duration_s: float = 3.0,
  overlap_duration_s: float = 0.0,
  speed: float = 1.0,
) -> AcousticFileEncodingResult:
  assert n_files > 0
  assert 0 <= overlap_duration_s < segment_duration_s
  np.random.seed(0)
  n_segments = get_n_segments_speed(
    duration_s, segment_duration_s, overlap_duration_s, speed
  )
  species_emb = np.random.random((n_files, n_segments, 1024)).astype(np.float32)
  species_masked = np.full((n_files, n_segments, 1024), False, dtype=bool)

  tensor = create_mock_tensor_one_unproc(species_emb, species_masked)

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
    model_fmax=15_000,
    model_fmin=0,
    model_path=Path("/model/path"),
    model_precision="fp32",
    model_sr=48_000,
    model_version="v2.4",
  )


def test_loading_a_second_file_leaves_the_first_result_untouched(
  tmp_path: Path,
) -> None:
  first = create_dummy_result(n_files=3, duration_s=20.0)
  second = create_dummy_result(n_files=2, duration_s=12.0)
  expected = saved_arrays(first)
  first.save(tmp_path / "first.npz", compress=False)
  second.save(tmp_path / "second.npz", compress=False)

  loaded_first = type(first).load(tmp_path / "first.npz")
  loaded_second = type(second).load(tmp_path / "second.npz")

  actual = saved_arrays(loaded_first)
  other = saved_arrays(loaded_second)
  assert actual.keys() == expected.keys()
  for key, array in expected.items():
    np.testing.assert_array_equal(actual[key], array, err_msg=key)
    # Equal values would hide a field the loader still stores on the class.
    assert actual[key] is not other[key], key


def test_memory_size_MiB() -> None:
  res = create_dummy_result()

  expected_size = (
    res._inputs.nbytes
    + res._input_durations.nbytes
    + res._segment_duration_s.nbytes
    + res._overlap_duration_s.nbytes
    + res._speed.nbytes
    + res._model_path.nbytes
    + res._model_version.nbytes
    + res._model_precision.nbytes
    + res._model_fmax.nbytes
    + res._model_fmin.nbytes
    + res._model_sr.nbytes
    + res._embeddings.nbytes
    + res._embeddings_masked.nbytes
    + res._unprocessable_inputs.nbytes
  ) / 1024**2

  assert res.memory_size_MiB == expected_size
