from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.acoustic_models.inference.predictions.prediction_result_py.test_prediction_to_structured_array import (  # noqa: E501
  create_file_prediction_result,
)
from birdnet_tests.helper import saved_arrays

pytestmark = pytest.mark.no_tf


def test_loading_a_second_file_leaves_the_first_result_untouched(
  tmp_path: Path,
) -> None:
  first = create_file_prediction_result(
    n_files=3,
    duration_s=54,
    top_k=5,
    segment_duration_s=3.0,
    overlap_duration_s=1.5,
  )
  second = create_file_prediction_result(
    n_files=2,
    duration_s=30,
    top_k=4,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
  )
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
  res = create_file_prediction_result(
    n_files=3,
    duration_s=54,
    top_k=5,
    segment_duration_s=3.0,
    overlap_duration_s=1.5,
    speed=1.0,
    unprocessable_files={0},
  )

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
    + res._species_probs.nbytes
    + res._species_ids.nbytes
    + res._species_list.nbytes
    + res._species_masked.nbytes
    + res._unprocessable_inputs.nbytes
  ) / 1024**2

  assert res.memory_size_MiB == expected_size
