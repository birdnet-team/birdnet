from pathlib import Path
from typing import Literal, overload

from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.base import (
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
  MODEL_VERSION_V2_4,
  MODEL_VERSIONS,
)


@overload
def load(  # type: ignore
  *,
  model_type: Literal["acoustic"] = ...,
  version: Literal["2.4"] = ...,
  backend: Literal["tf"] = ...,
  lang_id: str = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load(
  *,
  model_type: Literal["acoustic"] = ...,
  version: Literal["2.4"] = ...,
  backend: Literal["pb"] = ...,
  lang_id: str = ...,
) -> AcousticPBModelV2_4: ...


# @overload
# def load(
#   *,
#   model_type: Literal["acoustic"] = MODEL_TYPE_ACOUSTIC,
#   version: Literal["2.4"] = MODEL_VERSION_V2_4,
#   backend: Literal["tf"] = MODEL_BACKEND_TF,
#   device: Literal["CPU", "GPU"] = "CPU",
#   lang_id: str = "en_us",
# ) -> AcousticTFModelV2_4: ...


# @overload
# def load(
#   *,
#   model_type: Literal["acoustic"] = MODEL_TYPE_ACOUSTIC,
#   version: Literal["2.4"] = MODEL_VERSION_V2_4,
#   backend: Literal["pb"] = MODEL_BACKEND_PB,
#   device: Literal["CPU", "GPU"] = "CPU",
#   lang_id: str = "en_us",
# ) -> AcousticPBModelV2_4: ...


def load(
  *,
  model_type: MODEL_TYPES = MODEL_TYPE_ACOUSTIC,
  version: MODEL_VERSIONS = MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  lang_id: str = "en_us",
):
  if model_type == MODEL_TYPE_ACOUSTIC:
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        return AcousticTFModelV2_4.load_official(lang_id)
      else:
        assert backend == MODEL_BACKEND_PB
        return AcousticPBModelV2_4.load_official(lang_id)
    raise AssertionError()
  else:
    assert model_type == MODEL_TYPE_GEO
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        pass
      else:
        assert backend == MODEL_BACKEND_PB
    raise AssertionError()


def load_custom(
  model_path: Path,
  species_list: Path,
  model_type: MODEL_TYPES = MODEL_TYPE_ACOUSTIC,
  version: MODEL_VERSIONS = MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
):
  if model_type == MODEL_TYPE_ACOUSTIC:
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        return AcousticTFModelV2_4.load_custom(model_path, species_list)
      else:
        assert backend == MODEL_BACKEND_PB
    raise AssertionError()
  else:
    assert model_type == MODEL_TYPE_GEO
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        pass
      else:
        assert backend == MODEL_BACKEND_PB
    raise AssertionError()
