from os import PathLike
from pathlib import Path
from typing import Literal, overload

from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.base import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  GEO_MODEL_VERSIONS,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FLOAT16,
  MODEL_PRECISION_FLOAT32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
  VALID_ACOUSTIC_MODEL_VERSIONS,
  VALID_MODEL_BACKENDS,
  VALID_MODEL_PRECISIONS,
  ModelBase,
)
from birdnet.geo_models.base import GeoModelBase
from birdnet.helper import tf_installed


@overload
def load(  # type: ignore
  *,
  version: Literal["latest"] | ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["tf"] = ...,
  precision: MODEL_PRECISIONS = ...,
  lang_id: MODEL_LANGUAGES = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load(
  *,
  version: Literal["latest"] | ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["pb"] = ...,
  precision: Literal["fp32"] = ...,
  lang_id: MODEL_LANGUAGES = ...,
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
  version: Literal["latest"] | ACOUSTIC_MODEL_VERSIONS = "latest",
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  lang_id: MODEL_LANGUAGES = "en_us",
) -> AcousticModelBase:
  if version not in ["latest"] + VALID_ACOUSTIC_MODEL_VERSIONS:
    raise ValueError(
      f"Parameter 'version': Unsupported model version: {version}. Available versions are: {', '.join(VALID_ACOUSTIC_MODEL_VERSIONS)}."
    )

  if backend not in VALID_MODEL_BACKENDS:
    raise ValueError(
      f"Parameter 'backend': Unknown model backend: {backend}. Available backends are: {', '.join(VALID_MODEL_BACKENDS)}."
    )

  if precision not in VALID_MODEL_PRECISIONS:
    raise ValueError(
      f"Parameter 'precision': Unsupported model precision: {precision}. Currently supported precisions: {', '.join(VALID_MODEL_PRECISIONS)}."
    )

  if version == "latest":
    version = ACOUSTIC_MODEL_VERSION_V2_4

  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    from birdnet.acoustic_models.v2_4.base import AVAILABLE_LANGUAGES

    if lang_id not in AVAILABLE_LANGUAGES:
      raise ValueError(
        f"Parameter 'lang_id': Language '{lang_id}' is not supported by the model."
      )

    if backend == MODEL_BACKEND_TF:
      return AcousticTFModelV2_4.load_official(lang_id, precision)
    elif backend == MODEL_BACKEND_PB:
      if precision != MODEL_PRECISION_FLOAT32:
        raise ValueError(
          f"Parameter 'precision': Unsupported model precision for 'pb': {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
        )

      if not tf_installed():
        raise ValueError(
          "TensorFlow is not available. Cannot load custom 'pb' model. Install birdnet with [tf] option."
        )
      return AcousticPBModelV2_4.load_official(lang_id)
    else:
      raise AssertionError()
  else:
    raise AssertionError()


@overload
def load_custom(  # type: ignore
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  version: ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["tf"] = ...,
  precision: MODEL_PRECISIONS = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load_custom(
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  version: ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["pb"] = ...,
  precision: Literal["fp32"] = ...,
) -> AcousticPBModelV2_4: ...


def load_custom(
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  version: ACOUSTIC_MODEL_VERSIONS = ACOUSTIC_MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
) -> ModelBase:
  if version != ACOUSTIC_MODEL_VERSION_V2_4:
    raise ValueError(
      f"Parameter 'version': Unsupported model version: {version}. Available version is: {ACOUSTIC_MODEL_VERSION_V2_4}."
    )

  if backend not in (MODEL_BACKEND_TF, MODEL_BACKEND_PB):
    raise ValueError(
      f"Parameter 'backend': Unknown model backend: {backend}. Available backends are: {MODEL_BACKEND_TF}, {MODEL_BACKEND_PB}."
    )

  if precision not in (
    MODEL_PRECISION_INT8,
    MODEL_PRECISION_FLOAT16,
    MODEL_PRECISION_FLOAT32,
  ):
    raise ValueError(
      f"Parameter 'precision': Unsupported model precision: {precision}. Currently supported precisions: {MODEL_PRECISION_INT8}, {MODEL_PRECISION_FLOAT16}, {MODEL_PRECISION_FLOAT32}."
    )

  model_path = Path(model)
  if not model_path.is_file():
    raise ValueError(
      f"Parameter 'model_path': Model file '{model_path.absolute()}' does not exist!"
    )

  species_list_path = Path(species_list)
  if not species_list_path.is_file():
    raise ValueError(
      f"Parameter 'species_list': Species list file '{species_list_path.absolute()}' does not exist!"
    )

  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    if backend == MODEL_BACKEND_TF:
      return AcousticTFModelV2_4.load_custom(model_path, species_list_path, precision)
    elif backend == MODEL_BACKEND_PB:
      if precision != MODEL_PRECISION_FLOAT32:
        raise ValueError(
          f"Parameter 'precision': Unsupported model precision for 'pb': {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
        )

      if not tf_installed():
        raise ValueError(
          "TensorFlow is not available. Cannot load custom 'pb' model. Install birdnet with [tf] option."
        )
      return AcousticPBModelV2_4.load_custom(model_path, species_list_path)
    else:
      raise AssertionError()
  else:
    raise AssertionError()


def load_geo(
  *,
  version: Literal["latest"] | GEO_MODEL_VERSIONS = ACOUSTIC_MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  lang_id: str = "en_us",
) -> GeoModelBase:
  pass
