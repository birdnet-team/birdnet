"""The facts a model answers before loading are abstract on the base classes."""

import inspect

import pytest

from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.core.base import ModelBase
from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.geo.models.v3_0.model import GeoModelV3_0

pytestmark = pytest.mark.no_tf

_OFFICIAL_MODELS = [
  AcousticModelV2_4,
  AcousticModelV3_0,
  AcousticModelPerchV2,
  GeoModelV2_4,
  GeoModelV3_0,
]


def test_get_n_classes_is_abstract_on_the_model_base() -> None:
  assert "get_n_classes" in ModelBase.__abstractmethods__


def test_get_embeddings_dim_is_abstract_on_the_acoustic_base() -> None:
  assert "get_embeddings_dim" in AcousticModelBase.__abstractmethods__


@pytest.mark.parametrize("model", _OFFICIAL_MODELS, ids=lambda m: m.__name__)
def test_every_official_model_answers_without_loading(model: type[ModelBase]) -> None:
  assert not inspect.isabstract(model)
  assert model.get_n_classes() > 0
