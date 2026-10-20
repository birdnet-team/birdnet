"""Every result of a session shares the species arrays the session built once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.geo_models.inference.test_geo_session import (
  FakeGeoBackend,
  create_geo_session,
)

pytestmark = pytest.mark.no_tf


def test_runs_share_the_species_arrays(tmp_path: Path) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()
  FakeGeoBackend.configure(
    prediction_batch=np.array([[0.1, 0.8, 0.9]], dtype=np.float32),
    week_inputs=(1.0,),
  )

  with create_geo_session(model_path) as session:
    first = session.run(20, 50, week=4)
    second = session.run(-10, 120, week=30)

  assert first.species_list is second.species_list
  assert first.species_ids is second.species_ids
  assert not first.species_list.flags.writeable
  assert not first.species_ids.flags.writeable

  assert first.species_list.dtype == np.dtype("<U9")
  np.testing.assert_array_equal(
    first.species_list, ["species_a", "species_b", "species_c"]
  )
  assert first.species_ids.dtype == np.uint8
  np.testing.assert_array_equal(first.species_ids, [0, 1, 2])
