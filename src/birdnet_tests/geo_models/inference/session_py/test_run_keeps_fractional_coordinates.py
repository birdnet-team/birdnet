"""The result reports the coordinates it was asked about, fraction included."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.geo_models.inference.test_geo_session import (
  FakeGeoBackend,
  create_geo_session,
)

pytestmark = pytest.mark.no_tf


def test_result_keeps_the_fractional_coordinates(tmp_path: Path) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()
  FakeGeoBackend.configure(
    prediction_batch=np.array([[0.1, 0.8, 0.9]], dtype=np.float32),
    week_inputs=(1.0, 24.0, 48.0),
  )

  with create_geo_session(model_path) as session:
    result = session.run(42.5, -76.45, week=4)

  assert isinstance(result.latitude, float)
  assert isinstance(result.longitude, float)
  # The result stores the coordinates as float32.
  assert result.latitude == pytest.approx(42.5)
  assert result.longitude == pytest.approx(-76.45)
