from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.globals import (
  GEO_YEAR_ROUND_AGGREGATION_AVERAGE,
  GEO_YEAR_ROUND_AGGREGATION_MAX,
)


class FakeGeoBackend:
  configured_prediction_batch: ClassVar[np.ndarray] = np.empty((0, 0), dtype=np.float32)
  configured_week_inputs: ClassVar[tuple[float, ...]] = ()
  captured_batches: ClassVar[list[np.ndarray]] = []

  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
  ) -> None:
    self._model_path = model_path
    self._device_name = device_name
    self._half_precision = half_precision

  @classmethod
  def configure(
    cls,
    *,
    prediction_batch: np.ndarray,
    week_inputs: tuple[float, ...],
  ) -> None:
    cls.configured_prediction_batch = prediction_batch
    cls.configured_week_inputs = week_inputs
    cls.captured_batches = []

  def load(self) -> None:
    pass

  def unload(self) -> None:
    pass

  def predict(self, batch: np.ndarray) -> np.ndarray:
    type(self).captured_batches.append(batch.copy())
    return type(self).configured_prediction_batch.copy()

  def encode(self, batch: np.ndarray) -> np.ndarray:
    return batch

  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @property
  def n_species(self) -> int:
    return int(type(self).configured_prediction_batch.shape[1])

  @classmethod
  def precision(cls) -> str:
    return "fp32"

  @classmethod
  def name(cls) -> str:
    return "fake-geo"

  def copy_to_device(self, batch: np.ndarray) -> np.ndarray:
    return batch

  def copy_from_device(self, inference_result: np.ndarray) -> np.ndarray:
    return inference_result

  def half_precision(self, inference_result: np.ndarray) -> np.ndarray:
    return inference_result

  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]:
    return cls.configured_week_inputs


def create_geo_session(model_path: Path) -> GeoPredictionSession:
  return GeoPredictionSession(
    species_list=OrderedSet(["species_a", "species_b", "species_c"]),
    model_path=model_path,
    model_is_custom=False,
    model_version="3.0",
    model_backend_type=FakeGeoBackend,
    model_backend_custom_kwargs={},
    min_confidence=0.5,
    half_precision=False,
    device="CPU",
  )


@pytest.mark.parametrize(
  ("aggregation", "expected_probs"),
  [
    (
      GEO_YEAR_ROUND_AGGREGATION_MAX,
      np.array([0.5, 0.8, 0.9], dtype=np.float32),
    ),
    (
      GEO_YEAR_ROUND_AGGREGATION_AVERAGE,
      np.array([0.3, 0.53333336, 0.6666667], dtype=np.float32),
    ),
  ],
)
def test_run_week_none_batches_and_aggregates_year_round_predictions(
  tmp_path: Path,
  aggregation: str,
  expected_probs: np.ndarray,
) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()

  prediction_batch = np.array(
    [
      [0.1, 0.8, 0.4],
      [0.5, 0.2, 0.9],
      [0.3, 0.6, 0.7],
    ],
    dtype=np.float32,
  )
  week_inputs = (1.0, 24.0, 48.0)
  expected_samples = np.array(
    [
      [20.0, 50.0, 1.0],
      [20.0, 50.0, 24.0],
      [20.0, 50.0, 48.0],
    ],
    dtype=np.float32,
  )

  FakeGeoBackend.configure(
    prediction_batch=prediction_batch,
    week_inputs=week_inputs,
  )

  with create_geo_session(model_path) as session:
    result = session.run(
      20,
      50,
      week=None,
      year_round_aggregation=aggregation,
    )

  assert len(FakeGeoBackend.captured_batches) == 1
  np.testing.assert_array_equal(FakeGeoBackend.captured_batches[0], expected_samples)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == -1
  np.testing.assert_array_equal(result.species_probs, expected_probs)
  np.testing.assert_array_equal(result.species_masked, expected_probs < 0.5)
  np.testing.assert_array_equal(
    result.species_ids,
    np.array([0, 1, 2], dtype=np.uint8),
  )


def test_run_with_specific_week_uses_single_sample(tmp_path: Path) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()

  prediction_batch = np.array([[0.1, 0.8, 0.9]], dtype=np.float32)
  FakeGeoBackend.configure(
    prediction_batch=prediction_batch,
    week_inputs=(1.0, 24.0, 48.0),
  )

  with create_geo_session(model_path) as session:
    result = session.run(20, 50, week=12)

  # exactly one [lat, lon, week] sample is fed to the backend
  assert len(FakeGeoBackend.captured_batches) == 1
  np.testing.assert_array_equal(
    FakeGeoBackend.captured_batches[0],
    np.array([[20.0, 50.0, 12.0]], dtype=np.float32),
  )
  assert result.week == 12
  np.testing.assert_array_equal(result.species_probs, prediction_batch[0])
  np.testing.assert_array_equal(result.species_masked, prediction_batch[0] < 0.5)


def test_run_week_none_with_single_week_input_squeezes(tmp_path: Path) -> None:
  model_path = tmp_path / "geo-model.bin"
  model_path.touch()

  # A single year-round week input exercises the squeeze branch (no aggregation).
  prediction_batch = np.array([[0.2, 0.6, 0.95]], dtype=np.float32)
  FakeGeoBackend.configure(
    prediction_batch=prediction_batch,
    week_inputs=(1.0,),
  )

  with create_geo_session(model_path) as session:
    result = session.run(10, 30, week=None)

  assert len(FakeGeoBackend.captured_batches) == 1
  np.testing.assert_array_equal(
    FakeGeoBackend.captured_batches[0],
    np.array([[10.0, 30.0, 1.0]], dtype=np.float32),
  )
  assert result.week == -1
  np.testing.assert_array_equal(result.species_probs, prediction_batch[0])
