from os import PathLike
from typing import Literal, overload

from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.base import (
  MODEL_BACKENDS,
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FLOAT32,
  MODEL_PRECISIONS,
)
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4

@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> AcousticTFModelV2_4: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> AcousticPBModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: MODEL_BACKENDS,
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> AcousticPBModelV2_4 | AcousticTFModelV2_4: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> GeoTFModelV2_4: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> GeoPBModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: MODEL_BACKENDS,
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> GeoTFModelV2_4 | GeoPBModelV2_4: ...

# LOAD CUSTOM MODELS

@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> AcousticTFModelV2_4: ...

# @overload
# def load_custom(
#   model_type: Literal["acoustic"],
#   version: Literal["3.0"],
#   backend: Literal["tf"],
#   model: str | PathLike[str],
#   species_list: str | PathLike[str],
#   /,
#   *,
#   precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
#   check_validity: bool = True,
# ) -> AcousticTFModelV3_0: ...

@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> AcousticPBModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: MODEL_BACKENDS,
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> AcousticPBModelV2_4 | AcousticTFModelV2_4: ...
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> GeoTFModelV2_4: ...
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> GeoPBModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: MODEL_BACKENDS,
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> GeoTFModelV2_4 | GeoPBModelV2_4: ...
