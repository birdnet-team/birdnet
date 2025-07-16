from os import PathLike
from pathlib import Path
from typing import Literal, overload

from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.base import (
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_PRECISION_FLOAT16,
  MODEL_PRECISION_FLOAT32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
  MODEL_VERSION_V2_4,
  MODEL_VERSIONS,
  ModelBase,
)
from birdnet.helper import tf_installed


@overload
def load(  # type: ignore
  *,
  model_type: Literal["acoustic"] = ...,
  version: MODEL_VERSIONS = ...,
  backend: Literal["tf"] = ...,
  precision: MODEL_PRECISIONS = ...,
  lang_id: str = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load(
  *,
  model_type: Literal["acoustic"] = ...,
  version: MODEL_VERSIONS = ...,
  backend: Literal["pb"] = ...,
  precision: Literal["fp32"] = ...,
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
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  lang_id: str = "en_us",
) -> ModelBase:
  if model_type not in (MODEL_TYPE_ACOUSTIC, MODEL_TYPE_GEO):
    raise ValueError(
      f"Parameter 'model_type': Unknown model type: {model_type}. Available types are: {MODEL_TYPE_ACOUSTIC}, {MODEL_TYPE_GEO}."
    )

  if version != MODEL_VERSION_V2_4:
    raise ValueError(
      f"Parameter 'version': Unsupported model version: {version}. Available version is: {MODEL_VERSION_V2_4}."
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

  if model_type == MODEL_TYPE_ACOUSTIC:
    if version == MODEL_VERSION_V2_4:
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
  elif model_type == MODEL_TYPE_GEO:
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        raise NotImplementedError()
      elif backend == MODEL_BACKEND_PB:
        raise NotImplementedError()
      else:
        raise AssertionError()
    else:
      raise AssertionError()
  else:
    raise AssertionError()


@overload
def load_custom(  # type: ignore
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  model_type: Literal["acoustic"] = ...,
  version: MODEL_VERSIONS = ...,
  backend: Literal["tf"] = ...,
  precision: MODEL_PRECISIONS = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load_custom(
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  model_type: Literal["acoustic"] = ...,
  version: MODEL_VERSIONS = ...,
  backend: Literal["pb"] = ...,
  precision: Literal["fp32"] = ...,
) -> AcousticPBModelV2_4: ...


def load_custom(
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  model_type: MODEL_TYPES = MODEL_TYPE_ACOUSTIC,
  version: MODEL_VERSIONS = MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
) -> ModelBase:
  if model_type not in (MODEL_TYPE_ACOUSTIC, MODEL_TYPE_GEO):
    raise ValueError(
      f"Parameter 'model_type': Unknown model type: {model_type}. Available types are: {MODEL_TYPE_ACOUSTIC}, {MODEL_TYPE_GEO}."
    )

  if version != MODEL_VERSION_V2_4:
    raise ValueError(
      f"Parameter 'version': Unsupported model version: {version}. Available version is: {MODEL_VERSION_V2_4}."
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

  if model_type == MODEL_TYPE_ACOUSTIC:
    if version == MODEL_VERSION_V2_4:
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
  elif model_type == MODEL_TYPE_GEO:
    raise ValueError(
      "Parameter 'model_type': Custom geo models are currently not supported."
    )
  else:
    raise AssertionError()
