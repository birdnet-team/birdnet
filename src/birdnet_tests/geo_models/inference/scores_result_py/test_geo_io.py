import tempfile
from pathlib import Path

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet_tests.helper import saved_arrays


def get_dummy_result(
  latitude: float = 10.0,
  longitude: float = 20.0,
  week: int = 5,
  species_probs: np.ndarray | None = None,
) -> GeoPredictionResult:
  if species_probs is None:
    species_probs = np.array([0.1, 0.5, 0.9], dtype=np.float32)
  res = GeoPredictionResult(
    latitude=latitude,
    longitude=longitude,
    week=week,
    model_path=Path("/dummy/path"),
    model_version="2.4",
    model_precision="fp32",
    species_ids=np.array([1, 2, 3], dtype=np.uint16),
    species_probs=species_probs,
    species_masked=np.array([True, False, False], dtype=bool),
    species_list=OrderedSet(["species_a", "species_b", "species_c"]),
  )
  return res


def test_memory_size_MiB() -> None:
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

  assert res.memory_size_MiB == expected_size


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


@pytest.mark.no_tf
def test_loading_a_second_file_leaves_the_first_result_untouched(
  tmp_path: Path,
) -> None:
  first = get_dummy_result()
  second = get_dummy_result(
    latitude=42.5,
    longitude=-76.45,
    week=7,
    species_probs=np.array([0.2, 0.3, 0.4], dtype=np.float32),
  )
  expected = saved_arrays(first)
  first.save(tmp_path / "first.npz", compress=False)
  second.save(tmp_path / "second.npz", compress=False)

  loaded_first = GeoPredictionResult.load(tmp_path / "first.npz")
  loaded_second = GeoPredictionResult.load(tmp_path / "second.npz")

  actual = saved_arrays(loaded_first)
  other = saved_arrays(loaded_second)
  assert actual.keys() == expected.keys()
  for key, array in expected.items():
    np.testing.assert_array_equal(actual[key], array, err_msg=key)
    # Equal values would hide a field the loader still stores on the class.
    assert actual[key] is not other[key], key
