import tempfile
from pathlib import Path

import numpy as np
from ordered_set import OrderedSet

from birdnet.geo_models.inference.prediction_result import GeoPredictionResult


def get_dummy_result() -> GeoPredictionResult:
  res = GeoPredictionResult(
    latitude=10.0,
    longitude=20.0,
    week=5,
    model_path=Path("/dummy/path"),
    model_version="2.4",
    model_precision="fp32",
    species_ids=np.array([1, 2, 3], dtype=np.uint16),
    species_probs=np.array([0.1, 0.5, 0.9], dtype=np.float32),
    species_masked=np.array([True, False, False], dtype=bool),
    species_list=OrderedSet(["species_a", "species_b", "species_c"]),
  )
  return res


def test_memory_size_mb() -> None:
  res = get_dummy_result()

  expected_size = (
    res._species_ids.nbytes
    + res._species_probs.nbytes
    + res._species_masked.nbytes
    + res._species_list.nbytes
    + res._latitude.nbytes
    + res._longitude.nbytes
    + res._week.nbytes
    + res._model_path.nbytes
    + res._model_version.nbytes
    + res._model_precision.nbytes
  ) / 1024**2

  assert res.memory_size_mb == expected_size


def test_save_and_load_is_equal() -> None:
  reference = get_dummy_result()

  with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, mode="wb") as tmp_file:
    reference.save(tmp_file.name, compress=False)
  loaded = type(reference).load(tmp_file.name)
  tmp_file.close()

  np.testing.assert_array_equal(reference._model_path, loaded._model_path)
  np.testing.assert_array_equal(reference._model_version, loaded._model_version)
  np.testing.assert_array_equal(reference._model_precision, loaded._model_precision)

  np.testing.assert_array_equal(reference._latitude, loaded._latitude)
  np.testing.assert_array_equal(reference._longitude, loaded._longitude)
  np.testing.assert_array_equal(reference._week, loaded._week)

  np.testing.assert_array_equal(reference._species_list, loaded._species_list)
  np.testing.assert_array_equal(reference._species_ids, loaded._species_ids)
  np.testing.assert_array_equal(reference._species_probs, loaded._species_probs)
  np.testing.assert_array_equal(reference._species_masked, loaded._species_masked)
