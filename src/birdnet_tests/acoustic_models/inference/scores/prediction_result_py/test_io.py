import tempfile

import numpy as np

from birdnet_tests.acoustic_models.inference.scores.prediction_result_py.test_to_structured_array import (
  create_file_prediction_result,
)


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

  assert reference.inputs.shape == loaded.inputs.shape
  assert reference.input_durations.shape == loaded.input_durations.shape

  assert reference.model_path == loaded.model_path
  assert reference.model_version == loaded.model_version
  assert reference.model_precision == loaded.model_precision
  assert reference.model_sr == loaded.model_sr
  assert reference.model_fmin == loaded.model_fmin
  assert reference.model_fmax == loaded.model_fmax

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
