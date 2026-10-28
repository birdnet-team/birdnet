"""The append/replace boundary equals the public class count."""

import pytest

from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.custom_tflite import (
  _BASE_MODELS,
  CustomTFLiteLayout,
  describe_custom_tflite_layout,
)
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0

pytestmark = pytest.mark.no_tf

_OFFICIAL = {"2.4": AcousticModelV2_4, "3.0": AcousticModelV3_0}


def test_every_base_model_equals_the_public_class_count() -> None:
  assert {b.version: b.n_official_classes for b in _BASE_MODELS} == {
    version: model.get_n_classes() for version, model in _OFFICIAL.items()
  }


@pytest.mark.parametrize("model", list(_OFFICIAL.values()))
def test_the_public_class_count_is_the_append_boundary(
  model: type[AcousticModelBase],
) -> None:
  def mode(n_species: int) -> str:
    layout = CustomTFLiteLayout(
      input_idx=0,
      prediction_idx=5,
      embeddings_idx=4,
      embeddings_is_output=False,
      n_species=n_species,
      input_samples=model.get_segment_size_samples(),
      embeddings_dim=model.get_embeddings_dim(),
    )
    return describe_custom_tflite_layout(layout).mode

  assert mode(model.get_n_classes()) == "replace"
  assert mode(model.get_n_classes() + 1) == "append"
