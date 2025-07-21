from os import PathLike
from pathlib import Path
from typing import Literal, overload

from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.base import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FLOAT32,
  MODEL_PRECISIONS,
  VALID_ACOUSTIC_MODEL_VERSIONS,
  VALID_GEO_MODEL_VERSIONS,
  VALID_MODEL_BACKENDS,
  VALID_MODEL_PRECISIONS,
  ModelBase,
)
from birdnet.geo_models.base import GeoModelBase
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4


@overload
def load_acoustic_model(  # type: ignore
  *,
  version: Literal["latest"] | ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["tf"] = ...,
  precision: MODEL_PRECISIONS = ...,
  lang: MODEL_LANGUAGES = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load_acoustic_model(
  *,
  version: Literal["latest"] | ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["pb"] = ...,
  precision: Literal["fp32"] = ...,
  lang: MODEL_LANGUAGES = ...,
) -> AcousticPBModelV2_4: ...


# @overload
# def load(
#   *,
#   model_type: Literal["acoustic"] = MODEL_TYPE_ACOUSTIC,
#   version: Literal["2.4"] = MODEL_VERSION_V2_4,
#   backend: Literal["tf"] = MODEL_BACKEND_TF,
#   device: Literal["CPU", "GPU"] = "CPU",
#   lang: str = "en_us",
# ) -> AcousticTFModelV2_4: ...


# @overload
# def load(
#   *,
#   model_type: Literal["acoustic"] = MODEL_TYPE_ACOUSTIC,
#   version: Literal["2.4"] = MODEL_VERSION_V2_4,
#   backend: Literal["pb"] = MODEL_BACKEND_PB,
#   device: Literal["CPU", "GPU"] = "CPU",
#   lang: str = "en_us",
# ) -> AcousticPBModelV2_4: ...


def load_acoustic_model(
  *,
  version: Literal["latest"] | ACOUSTIC_MODEL_VERSIONS = "latest",
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = "en_us",
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
    from birdnet.translations import AVAILABLE_LANGUAGES_V2_4

    if lang not in AVAILABLE_LANGUAGES_V2_4:
      raise ValueError(
        f"Parameter 'lang': Language '{lang}' is not supported by the model."
      )

    if backend == MODEL_BACKEND_TF:
      return AcousticTFModelV2_4.load_official(lang, precision)
    elif backend == MODEL_BACKEND_PB:
      if precision != MODEL_PRECISION_FLOAT32:
        raise ValueError(
          f"Parameter 'precision': Unsupported model precision for 'pb': {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
        )

      return AcousticPBModelV2_4.load_official(lang)
    else:
      raise AssertionError()
  else:
    raise AssertionError()


@overload
def load_custom_acoustic_model(  # type: ignore
  model: str | PathLike[str] = ...,
  species_list: str | PathLike[str] = ...,
  version: ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["tf"] = ...,
  precision: MODEL_PRECISIONS = ...,
  check_validity: bool = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load_custom_acoustic_model(
  model: str | PathLike[str] = ...,
  species_list: str | PathLike[str] = ...,
  version: ACOUSTIC_MODEL_VERSIONS = ...,
  backend: Literal["pb"] = ...,
  precision: Literal["fp32"] = ...,
  check_validity: bool = ...,
) -> AcousticPBModelV2_4: ...


def load_custom_acoustic_model(  # type: ignore
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  version: ACOUSTIC_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  check_validity: bool = True,
) -> ModelBase:
  if version not in VALID_ACOUSTIC_MODEL_VERSIONS:
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

  species_list_path = Path(species_list)
  if not species_list_path.is_file():
    raise ValueError(
      f"Parameter 'species_list': Species list file '{species_list_path.absolute()}' does not exist!"
    )

  model_path = Path(model)

  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    if backend == MODEL_BACKEND_TF:
      if not model_path.is_file():
        raise ValueError(
          f"Parameter 'model_path': Model file '{model_path.absolute()}' does not exist!"
        )

      return AcousticTFModelV2_4.load_custom(
        model_path, species_list_path, precision, check_validity
      )
    elif backend == MODEL_BACKEND_PB:
      if precision != MODEL_PRECISION_FLOAT32:
        raise ValueError(
          f"Parameter 'precision': Unsupported model precision for 'pb': {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
        )

      if not model_path.is_dir():
        raise ValueError(
          f"Parameter 'model_path': Model directory '{model_path.absolute()}' does not exist!"
        )

      return AcousticPBModelV2_4.load_custom(
        model_path, species_list_path, check_validity
      )
    else:
      raise AssertionError()
  else:
    raise AssertionError()


def load_geo_model(
  *,
  version: Literal["latest"] | GEO_MODEL_VERSIONS = GEO_MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  lang: MODEL_LANGUAGES = "en_us",
) -> GeoModelBase:
  if version not in ["latest"] + VALID_GEO_MODEL_VERSIONS:
    raise ValueError(
      f"Parameter 'version': Unsupported model version: {version}. Available versions are: {', '.join(VALID_GEO_MODEL_VERSIONS)}."
    )

  if backend not in VALID_MODEL_BACKENDS:
    raise ValueError(
      f"Parameter 'backend': Unknown model backend: {backend}. Available backends are: {', '.join(VALID_MODEL_BACKENDS)}."
    )

  if version == "latest":
    version = GEO_MODEL_VERSION_V2_4

  if version == GEO_MODEL_VERSION_V2_4:
    from birdnet.translations import AVAILABLE_LANGUAGES_V2_4

    if lang not in AVAILABLE_LANGUAGES_V2_4:
      raise ValueError(
        f"Parameter 'lang': Language '{lang}' is not supported by the model."
      )

    if backend == MODEL_BACKEND_TF:
      return GeoTFModelV2_4.load_official(lang)
    elif backend == MODEL_BACKEND_PB:
      return GeoPBModelV2_4.load_official(lang)
    else:
      raise AssertionError()
  else:
    raise AssertionError()
