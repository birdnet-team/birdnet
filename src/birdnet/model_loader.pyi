from os import PathLike
from typing import Literal, overload

from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.globals import (
  LIBRARY_TF,
  LIBRARY_TYPES,
  MODEL_BACKENDS,
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)

@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
  library: LIBRARY_TYPES = LIBRARY_TF,
) -> AcousticTFModelV2_4: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
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
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> AcousticPBModelV2_4 | AcousticTFModelV2_4: ...

# if new versions are added, add this overload (also on the other places)
# @overload
# def load(
#   model_type: Literal["acoustic"],
#   version: ACOUSTIC_MODEL_VERSIONS,
#   backend: MODEL_BACKENDS,
#   /,
#   *,
#   precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
#   lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
# ) -> AcousticModelBase: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
  library: LIBRARY_TYPES = LIBRARY_TF,
) -> GeoTFModelV2_4: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
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
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
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
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  check_validity: bool = True,
  library: LIBRARY_TYPES = LIBRARY_TF,
) -> AcousticTFModelV2_4: ...
@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
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
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
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
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
  library: LIBRARY_TYPES = LIBRARY_TF,
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
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
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
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> GeoTFModelV2_4 | GeoPBModelV2_4: ...
