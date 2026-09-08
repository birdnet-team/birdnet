from os import PathLike
from typing import Literal, overload

from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.globals import (
  CUSTOM_CLASSIFIER_TYPES,
  CUSTOM_PB_IS_RAVEN_DEFAULT,
  LIBRARY_TF_DEFAULT,
  LIBRARY_TYPES,
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES_V2_4,
  MODEL_LANGUAGES_V3_0,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)

def load_perch_v2(device: Literal["CPU", "GPU"] = "CPU") -> AcousticModelPerchV2: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V2_4 = MODEL_LANGUAGE_EN_US,
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
) -> AcousticModelV2_4: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V2_4 = MODEL_LANGUAGE_EN_US,
) -> AcousticModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["tf", "pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V2_4 = MODEL_LANGUAGE_EN_US,
) -> AcousticModelV2_4: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["tf"],
  /,
  *,
  precision: Literal["fp32", "fp16"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
) -> AcousticModelV3_0: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> AcousticModelV3_0: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["pt"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> AcousticModelV3_0: ...
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["onnx"],
  /,
  *,
  precision: Literal["fp32", "fp16"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> AcousticModelV3_0: ...

# NOTE: to see "tf", "pb", "pt" and "onnx" overloads in the IDE
@overload
def load(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["tf", "pb", "pt", "onnx"],
  /,
  *,
  precision: Literal["fp32", "fp16"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> AcousticModelV3_0: ...

@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["tf"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V2_4 = MODEL_LANGUAGE_EN_US,
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
) -> GeoModelV2_4: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V2_4 = MODEL_LANGUAGE_EN_US,
) -> GeoModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["tf", "pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V2_4 = MODEL_LANGUAGE_EN_US,
) -> GeoModelV2_4: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["tf"],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
) -> GeoModelV3_0: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["pb"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> GeoModelV3_0: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["pt"],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> GeoModelV3_0: ...
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["onnx"],
  /,
  *,
  precision: Literal["fp32", "fp16"] = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> GeoModelV3_0: ...

# NOTE: to see "tf", "pb", "pt" and "onnx" overloads in the IDE
@overload
def load(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["tf", "pb", "pt", "onnx"],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  lang: MODEL_LANGUAGES_V3_0 = MODEL_LANGUAGE_EN_US,
) -> GeoModelV3_0: ...

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
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
  classifier_type: CUSTOM_CLASSIFIER_TYPES = "replace",
) -> AcousticModelV2_4: ...
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
  is_raven: bool = CUSTOM_PB_IS_RAVEN_DEFAULT,
) -> AcousticModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["2.4"],
  backend: Literal["tf", "pb"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> AcousticModelV2_4: ...
@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["pt"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> AcousticModelV3_0: ...
@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["onnx"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> AcousticModelV3_0: ...

# NOTE: to see "pt" and "onnx" overloads in the IDE
@overload
def load_custom(
  model_type: Literal["acoustic"],
  version: Literal["3.0"],
  backend: Literal["pt", "onnx"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> AcousticModelV3_0: ...
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
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
) -> GeoModelV2_4: ...
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
) -> GeoModelV2_4: ...

# NOTE: to see "tf" and "pb" overloads in the IDE
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["2.4"],
  backend: Literal["tf", "pb"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> GeoModelV2_4: ...
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["tf"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  check_validity: bool = True,
  library: LIBRARY_TYPES = LIBRARY_TF_DEFAULT,
) -> GeoModelV3_0: ...
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["pb"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> GeoModelV3_0: ...
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["pt"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> GeoModelV3_0: ...
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["onnx"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: Literal["fp32", "fp16"] = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> GeoModelV3_0: ...

# NOTE: to see "tf", "pb", "pt" and "onnx" overloads in the IDE
@overload
def load_custom(
  model_type: Literal["geo"],
  version: Literal["3.0"],
  backend: Literal["tf", "pb", "pt", "onnx"],
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FP32,
  check_validity: bool = True,
) -> GeoModelV3_0: ...
