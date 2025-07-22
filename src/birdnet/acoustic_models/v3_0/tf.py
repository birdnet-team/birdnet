from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic_models.v3_0.base import AcousticModelBaseV3_0
from birdnet.base import MODEL_PRECISIONS


class AcousticTFModelV3_0(AcousticModelBaseV3_0):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)
