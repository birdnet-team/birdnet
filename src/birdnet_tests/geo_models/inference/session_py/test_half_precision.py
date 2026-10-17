"""``half_precision`` decides the dtype of the scores a geo session returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.geo.inference.session import GeoPredictionSession
from birdnet_tests.geo_models.inference.test_geo_session import FakeGeoBackend

pytestmark = pytest.mark.no_tf


class CastingFakeGeoBackend(FakeGeoBackend):
  """Casts whenever asked, as ``PBBackend`` does: the session holds the flag."""

  def half_precision(self, inference_result: np.ndarray) -> np.ndarray:
    return inference_result.astype(np.float16)


def _session(model_path: Path, half_precision: bool) -> GeoPredictionSession:
  return GeoPredictionSession(
    species_list=OrderedSet(["species_a", "species_b", "species_c"]),
    model_path=model_path,
    model_is_custom=False,
    model_version="3.0",
    model_backend_type=CastingFakeGeoBackend,
    model_backend_custom_kwargs={},
    min_confidence=0.5,
    half_precision=half_precision,
    device="CPU",
  )


@pytest.mark.parametrize(
  ("half_precision", "expected_dtype"),
  [(True, np.float16), (False, np.float32)],
)
def test_species_probs_dtype_follows_half_precision(
  tmp_path: Path, half_precision: bool, expected_dtype: type
) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()
  CastingFakeGeoBackend.configure(
    prediction_batch=np.array([[0.1, 0.8, 0.9]], dtype=np.float32),
    week_inputs=(1.0, 24.0, 48.0),
  )

  with _session(model_path, half_precision) as session:
    result = session.run(20, 50, week=12)

  assert result.species_probs.dtype == expected_dtype
  np.testing.assert_allclose(result.species_probs, [0.1, 0.8, 0.9], atol=1e-3)
