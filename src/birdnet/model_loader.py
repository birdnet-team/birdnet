from os import PathLike
from pathlib import Path
from typing import cast

from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.base import ModelBase
from birdnet.geo_models.base import GeoModelBase
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  LIBRARY_TF,
  LIBRARY_TYPES,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
  VALID_ACOUSTIC_MODEL_VERSIONS,
  VALID_GEO_MODEL_VERSIONS,
  VALID_LIBRARY_TYPES,
  VALID_MODEL_BACKENDS,
  VALID_MODEL_LANGUAGES,
  VALID_MODEL_PRECISIONS,
  VALID_MODEL_TYPES,
)
from birdnet.helper import check_protobuf_model_files_exist


def _check_is_valid_model_type(model_type: str) -> MODEL_TYPES:
  if model_type not in VALID_MODEL_TYPES:
    raise ValueError(
      f"Unknown model type: {model_type}. Supported types are: {', '.join(VALID_MODEL_TYPES)}."
    )
  return cast(MODEL_TYPES, model_type)


def _check_is_valid_acoustic_model_version(version: str) -> ACOUSTIC_MODEL_VERSIONS:
  if version not in VALID_ACOUSTIC_MODEL_VERSIONS:
    raise ValueError(
      f"Unsupported model version: {version}. Available versions are: {', '.join(VALID_ACOUSTIC_MODEL_VERSIONS)}."
    )
  return cast(ACOUSTIC_MODEL_VERSIONS, version)


def _check_is_valid_geo_model_version(version: str) -> GEO_MODEL_VERSIONS:
  if version not in VALID_GEO_MODEL_VERSIONS:
    raise ValueError(
      f"Unsupported model version: {version}. Available versions are: {', '.join(VALID_GEO_MODEL_VERSIONS)}."
    )
  return cast(GEO_MODEL_VERSIONS, version)


def _check_is_valid_backend(backend: str) -> MODEL_BACKENDS:
  if backend not in VALID_MODEL_BACKENDS:
    raise ValueError(
      f"Unknown model backend: {backend}. Available backends are: {', '.join(VALID_MODEL_BACKENDS)}."
    )
  return cast(MODEL_BACKENDS, backend)


def _check_is_valid_precision(precision: str) -> MODEL_PRECISIONS:
  if precision not in VALID_MODEL_PRECISIONS:
    raise ValueError(
      f"Unsupported model precision: {precision}. Currently supported precisions: {', '.join(VALID_MODEL_PRECISIONS)}."
    )
  return cast(MODEL_PRECISIONS, precision)


def _check_is_valid_language(lang: str) -> MODEL_LANGUAGES:
  if lang not in VALID_MODEL_LANGUAGES:
    raise ValueError(
      f"Language '{lang}' is not supported by the model. Available languages are: {', '.join(VALID_MODEL_LANGUAGES)}."
    )
  return cast(MODEL_LANGUAGES, lang)


def _check_is_valid_species_list_path(species_list: str | PathLike[str]) -> Path:
  species_list = Path(species_list)
  if not species_list.is_file():
    raise ValueError(f"Species list file '{species_list.absolute()}' does not exist!")
  return species_list


def _check_is_valid_path(path: str | PathLike[str]) -> Path:
  path = Path(path)
  if not path.exists():
    raise ValueError(f"Path '{path.absolute()}' does not exist!")
  return path


def _check_is_valid_pb_model_folder(folder_path: Path) -> None:
  if not folder_path.is_dir():
    raise ValueError(f"Model folder '{folder_path.absolute()}' does not exist!")
  if not check_protobuf_model_files_exist(folder_path):
    raise ValueError(
      f"Model folder '{folder_path.absolute()}' does not contain valid protobuf model files!"
    )


def _check_is_valid_tf_file(model_path: Path) -> None:
  if not model_path.is_file():
    raise ValueError(f"Model file '{model_path.absolute()}' does not exist!")
  if not model_path.suffix == ".tflite":
    raise ValueError(
      f"Model file '{model_path.absolute()}' is not a valid TFLite model file!"
    )


def _check_is_valid_library(library: str) -> LIBRARY_TYPES:
  if library not in VALID_LIBRARY_TYPES:
    raise ValueError(
      f"Unsupported TensorFlow library: {library}. Supported libraries are:  {', '.join(VALID_LIBRARY_TYPES)}."
    )
  return cast(LIBRARY_TYPES, library)


def _check_allowed_kwargs(model_kwargs: dict, allowed: set[str] | None) -> None:
  if allowed is None:
    not_allowed = set(model_kwargs.keys())
  else:
    not_allowed = set(model_kwargs.keys()) - allowed
  if len(not_allowed) > 0:
    raise ValueError(f"Unexpected keyword arguments: {', '.join(not_allowed)}. ")


def load(
  model_type: str,
  version: str,
  backend: str,
  /,
  *,
  precision: str = MODEL_PRECISION_FP32,
  lang: str = MODEL_LANGUAGE_EN_US,
  **model_kwargs: object,
) -> ModelBase:
  model_type = _check_is_valid_model_type(model_type)
  backend = _check_is_valid_backend(backend)
  precision = _check_is_valid_precision(precision)
  lang = _check_is_valid_language(lang)

  if model_type == MODEL_TYPE_ACOUSTIC:
    version = _check_is_valid_acoustic_model_version(version)
    return _load_acoustic_model(
      version=version,
      backend=backend,
      precision=precision,
      lang=lang,
      **model_kwargs,
    )
  elif model_type == MODEL_TYPE_GEO:
    version = _check_is_valid_geo_model_version(version)
    return _load_geo_model(
      version=version,
      backend=backend,
      precision=precision,
      lang=lang,
      **model_kwargs,
    )
  else:
    raise AssertionError()


def _load_acoustic_model(
  version: ACOUSTIC_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
  **model_kwargs: object,
) -> AcousticModelBase:
  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    return _load_acoustic_model_V2_4(
      backend=backend,
      precision=precision,
      lang=lang,
      **model_kwargs,
    )
  else:
    raise AssertionError()


def _load_geo_model(
  version: GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
  **model_kwargs: object,
) -> GeoModelBase:
  if version == GEO_MODEL_VERSION_V2_4:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    return _load_geo_model_V2_4(backend, lang, **model_kwargs)
  else:
    raise AssertionError()


def _load_acoustic_model_V2_4(
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
  **model_kwargs: object,
) -> AcousticModelBaseV2_4:
  if backend == MODEL_BACKEND_TF:
    _check_allowed_kwargs(model_kwargs, {"library"})
    library = cast(str, model_kwargs.get("library", LIBRARY_TF))
    library = _check_is_valid_library(library)
    return AcousticTFModelV2_4.load(lang, precision, library)
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for acoustic pb model: {precision}. Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    _check_allowed_kwargs(model_kwargs, None)

    return AcousticPBModelV2_4.load(lang)
  else:
    raise AssertionError()


def _load_geo_model_V2_4(
  backend: MODEL_BACKENDS,
  lang: MODEL_LANGUAGES,
  **model_kwargs: object,
) -> GeoModelBase:
  if backend == MODEL_BACKEND_TF:
    _check_allowed_kwargs(model_kwargs, {"library"})
    library = cast(str, model_kwargs.get("library", LIBRARY_TF))
    library = _check_is_valid_library(library)
    return GeoTFModelV2_4.load(lang, library)
  elif backend == MODEL_BACKEND_PB:
    _check_allowed_kwargs(model_kwargs, None)
    return GeoPBModelV2_4.load(lang)
  else:
    raise AssertionError()


def load_custom(
  model_type: str,
  version: str,
  backend: str,
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: str = MODEL_PRECISION_FP32,
  check_validity: bool = True,
  **model_kwargs: object,
) -> ModelBase:
  model_type = _check_is_valid_model_type(model_type)
  backend = _check_is_valid_backend(backend)
  model = _check_is_valid_path(model)
  species_list = _check_is_valid_species_list_path(species_list)
  precision = _check_is_valid_precision(precision)

  if model_type == MODEL_TYPE_ACOUSTIC:
    version = _check_is_valid_acoustic_model_version(version)
    return _load_custom_acoustic_model(
      version=version,
      backend=backend,
      precision=precision,
      model=model,
      species_list=species_list,
      check_validity=check_validity,
      **model_kwargs,
    )
  elif model_type == MODEL_TYPE_GEO:
    version = _check_is_valid_geo_model_version(version)
    return _load_custom_geo_model(
      version=version,
      backend=backend,
      model=model,
      precision=precision,
      species_list=species_list,
      check_validity=check_validity,
      **model_kwargs,
    )
  else:
    raise AssertionError()


def _load_custom_acoustic_model(
  version: ACOUSTIC_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  model: Path,
  species_list: Path,
  check_validity: bool,
  **model_kwargs: object,
) -> AcousticModelBase:
  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    return _load_custom_acoustic_model_V2_4(
      backend=backend,
      precision=precision,
      species_list=species_list,
      model=model,
      check_validity=check_validity,
      **model_kwargs,
    )
  else:
    raise AssertionError()


def _load_custom_geo_model(
  version: GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  model: Path,
  precision: MODEL_PRECISIONS,
  species_list: Path,
  check_validity: bool,
  **model_kwargs: object,
) -> GeoModelBase:
  if version == GEO_MODEL_VERSION_V2_4:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    return _load_custom_geo_model_V2_4(
      backend, model, species_list, check_validity, **model_kwargs
    )
  else:
    raise AssertionError()


def _load_custom_acoustic_model_V2_4(
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  model: Path,
  species_list: Path,
  check_validity: bool,
  **model_kwargs: object,
) -> AcousticModelBaseV2_4:
  if backend == MODEL_BACKEND_TF:
    _check_is_valid_tf_file(model)
    _check_allowed_kwargs(model_kwargs, {"library"})
    library = cast(str, model_kwargs.get("library", LIBRARY_TF))
    library = _check_is_valid_library(library)

    return AcousticTFModelV2_4.load_custom(
      model, species_list, precision, check_validity, library
    )
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for acoustic pb model: {precision}. Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    _check_is_valid_pb_model_folder(model)
    _check_allowed_kwargs(model_kwargs, None)
    return AcousticPBModelV2_4.load_custom(model, species_list, check_validity)
  else:
    raise AssertionError()


def _load_custom_geo_model_V2_4(
  backend: MODEL_BACKENDS,
  model: Path,
  species_list: Path,
  check_validity: bool,
  **model_kwargs: object,
) -> GeoModelBase:
  if backend == MODEL_BACKEND_TF:
    _check_is_valid_tf_file(model)
    _check_allowed_kwargs(model_kwargs, {"library"})
    library = cast(str, model_kwargs.get("library", LIBRARY_TF))
    library = _check_is_valid_library(library)
    return GeoTFModelV2_4.load_custom(model, species_list, check_validity, library)
  elif backend == MODEL_BACKEND_PB:
    _check_is_valid_pb_model_folder(model)
    _check_allowed_kwargs(model_kwargs, None)
    return GeoPBModelV2_4.load_custom(model, species_list, check_validity)
  else:
    raise AssertionError()
