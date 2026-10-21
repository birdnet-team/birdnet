from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.geo.models.base import GeoModelBase
from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.globals import GEO_YEAR_ROUND_AGGREGATION_AVERAGE
from birdnet_tests.geo_models.inference.test_geo_session import FakeGeoBackend

pytestmark = pytest.mark.no_tf

MODEL_CLASSES = [GeoModelV2_4, GeoModelV3_0]
SPECIES = ["a", "b", "c"]
WEEK_INPUTS = (1.0, 24.0, 48.0)
YEAR_ROUND_BATCH = np.array(
  [
    [0.1, 0.8, 0.4],
    [0.5, 0.2, 0.9],
    [0.3, 0.6, 0.7],
  ],
  dtype=np.float32,
)


class RecordingGeoBackend(FakeGeoBackend):
  constructed_with: ClassVar[list[dict[str, Any]]] = []

  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
    **backend_kwargs: Any,  # noqa: ANN401
  ) -> None:
    type(self).constructed_with.append(
      {
        "model_path": model_path,
        "device_name": device_name,
        "half_precision": half_precision,
        **backend_kwargs,
      }
    )
    super().__init__(model_path, device_name, half_precision)


def _load(
  tmp_path: Path,
  model_class: type[GeoModelBase],
  prediction_batch: np.ndarray,
  backend_kwargs: dict[str, Any] | None = None,
) -> GeoModelBase:
  model_path = tmp_path / "model.bin"
  model_path.touch()
  RecordingGeoBackend.configure(
    prediction_batch=prediction_batch, week_inputs=WEEK_INPUTS
  )
  RecordingGeoBackend.constructed_with = []
  return model_class.load(
    model_path,
    OrderedSet(SPECIES),
    RecordingGeoBackend,
    backend_kwargs or {},
  )


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_forwards_the_year_round_options(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(tmp_path, model_class, YEAR_ROUND_BATCH)

  result = model.predict(
    20,
    50,
    week=None,
    year_round_aggregation=GEO_YEAR_ROUND_AGGREGATION_AVERAGE,
    min_confidence=0.55,
  )

  np.testing.assert_allclose(
    result.species_probs, [0.3, 0.53333336, 0.6666667], rtol=1e-6
  )
  assert result.species_masked.tolist() == [True, True, False]
  assert result.week == -1
  assert result.model_version == model_class.get_version()


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_defaults_to_the_year_round_maximum(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(tmp_path, model_class, YEAR_ROUND_BATCH)

  result = model.predict(20, 50)

  np.testing.assert_array_equal(result.species_probs, YEAR_ROUND_BATCH.max(axis=0))
  assert result.week == -1


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_forwards_the_week_and_masks_below_the_default_confidence(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  # 0.02 and 0.04 straddle the default min_confidence of 0.03.
  model = _load(tmp_path, model_class, np.array([[0.02, 0.04, 0.9]], dtype=np.float32))

  result = model.predict(20, 50, week=12)

  np.testing.assert_array_equal(
    RecordingGeoBackend.captured_batches[0],
    np.array([[20.0, 50.0, 12.0]], dtype=np.float32),
  )
  assert result.week == 12
  assert result.species_masked.tolist() == [True, False, False]


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_session_defaults(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(tmp_path, model_class, np.array([[0.02, 0.04, 0.9]], dtype=np.float32))

  with model.predict_session() as session:
    result = session.run(20, 50, week=12)

  assert result.species_masked.tolist() == [True, False, False]
  assert RecordingGeoBackend.constructed_with[0]["device_name"] == "CPU"
  assert RecordingGeoBackend.constructed_with[0]["half_precision"] is False


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_labels_the_scores_with_the_models_species(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(tmp_path, model_class, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

  result = model.predict(20, 50, week=1)

  assert result.species_list.tolist() == SPECIES


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_releases_the_backend(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(tmp_path, model_class, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

  model.predict(20, 50, week=1)

  assert RecordingGeoBackend.unload_calls == 1


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_builds_the_backend_from_the_model_and_the_call(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(
    tmp_path,
    model_class,
    np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
    backend_kwargs={"option": "value"},
  )

  model.predict(20, 50, week=1)
  model.predict(20, 50, week=1, half_precision=True, device="GPU:1")

  assert RecordingGeoBackend.constructed_with == [
    {
      "model_path": model.model_path,
      "device_name": "CPU",
      "half_precision": False,
      "option": "value",
    },
    {
      "model_path": model.model_path,
      "device_name": "GPU:1",
      "half_precision": True,
      "option": "value",
    },
  ]


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_predict_takes_the_location_positionally_and_the_rest_by_keyword(
  tmp_path: Path, model_class: type[GeoModelBase]
) -> None:
  model = _load(tmp_path, model_class, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

  with pytest.raises(TypeError):
    model.predict(latitude=20, longitude=50)  # type: ignore[call-arg]
  with pytest.raises(TypeError):
    model.predict(20, 50, 12)  # type: ignore[misc]
