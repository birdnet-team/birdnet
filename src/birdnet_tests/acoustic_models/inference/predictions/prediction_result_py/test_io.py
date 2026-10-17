import tempfile
from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.acoustic_models.inference.predictions.prediction_result_py.test_prediction_to_structured_array import (  # noqa: E501
  create_file_prediction_result,
)
from birdnet_tests.helper import saved_arrays


def test_save_and_load_is_equal() -> None:
  reference = create_file_prediction_result(
    n_files=3,
    duration_s=54,
    top_k=5,
    segment_duration_s=3.0,
    overlap_duration_s=1.5,
    speed=1.0,
  )

  with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, mode="wb") as tmp_file:
    reference.save(tmp_file.name, compress=False)
  loaded = type(reference).load(tmp_file.name)
  tmp_file.close()

  np.testing.assert_array_equal(reference._model_path, loaded._model_path)
  np.testing.assert_array_equal(reference._model_version, loaded._model_version)
  np.testing.assert_array_equal(reference._model_precision, loaded._model_precision)
  np.testing.assert_array_equal(reference._model_sr, loaded._model_sr)
  np.testing.assert_array_equal(reference._model_fmin, loaded._model_fmin)
  np.testing.assert_array_equal(reference._model_fmax, loaded._model_fmax)

  np.testing.assert_array_equal(reference._inputs, loaded._inputs)
  np.testing.assert_array_equal(reference._input_durations, loaded._input_durations)

  np.testing.assert_array_equal(
    reference._segment_duration_s, loaded._segment_duration_s
  )
  np.testing.assert_array_equal(
    reference._overlap_duration_s, loaded._overlap_duration_s
  )
  np.testing.assert_array_equal(reference._speed, loaded._speed)

  np.testing.assert_array_equal(reference._species_list, loaded._species_list)
  np.testing.assert_array_equal(reference._species_ids, loaded._species_ids)
  np.testing.assert_array_equal(reference._species_probs, loaded._species_probs)
  np.testing.assert_array_equal(reference._species_masked, loaded._species_masked)
  np.testing.assert_array_equal(
    reference._unprocessable_inputs, loaded._unprocessable_inputs
  )


@pytest.mark.no_tf
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
