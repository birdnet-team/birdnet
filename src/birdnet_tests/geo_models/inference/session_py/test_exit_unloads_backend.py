"""Leaving a geo session unloads its backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.geo_models.inference.test_geo_session import (
  FakeGeoBackend,
  create_geo_session,
)

pytestmark = pytest.mark.no_tf


def test_exit_unloads_the_backend_once(tmp_path: Path) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()
  FakeGeoBackend.configure(
    prediction_batch=np.array([[0.1, 0.8, 0.9]], dtype=np.float32),
    week_inputs=(1.0,),
  )

  with create_geo_session(model_path) as session:
    session.run(20, 50, week=4)
    assert FakeGeoBackend.unload_calls == 0

  assert FakeGeoBackend.unload_calls == 1
