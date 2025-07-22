from os import PathLike
from pathlib import Path
from typing import Literal, cast

from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.v2_4.base import AcousticModelBaseV2_4
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
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FLOAT32,
  MODEL_PRECISIONS,
  VALID_ACOUSTIC_MODEL_VERSIONS,
  VALID_GEO_MODEL_VERSIONS,
  VALID_MODEL_BACKENDS,
  VALID_MODEL_LANGUAGES,
  VALID_MODEL_PRECISIONS,
  VALID_MODEL_TYPES,
  ModelBase,
)
from birdnet.geo_models.base import GeoModelBase
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.helper import check_protobuf_model_files_exist


def _check_is_valid_model_type(model_type: str) -> None:
  if model_type not in VALID_MODEL_TYPES:
    raise ValueError(
      f"Unknown model type: {model_type}. Supported types are: {', '.join(VALID_MODEL_TYPES)}."
    )


def _check_is_valid_acoustic_model_version(version: str) -> None:
  if version not in VALID_ACOUSTIC_MODEL_VERSIONS:
    raise ValueError(
      f"Unsupported model version: {version}. Available versions are: {', '.join(VALID_ACOUSTIC_MODEL_VERSIONS)}."
    )


def _check_is_valid_geo_model_version(version: str) -> None:
  if version not in VALID_GEO_MODEL_VERSIONS:
    raise ValueError(
      f"Unsupported model version: {version}. Available versions are: {', '.join(VALID_GEO_MODEL_VERSIONS)}."
    )


def _check_is_valid_backend(backend: str) -> None:
  if backend not in VALID_MODEL_BACKENDS:
    raise ValueError(
      f"Unknown model backend: {backend}. Available backends are: {', '.join(VALID_MODEL_BACKENDS)}."
    )


def _check_is_valid_precision(precision: str) -> None:
  if precision not in VALID_MODEL_PRECISIONS:
    raise ValueError(
      f"Unsupported model precision: {precision}. Currently supported precisions: {', '.join(VALID_MODEL_PRECISIONS)}."
    )


def _check_is_valid_language(lang: str) -> None:
  if lang not in VALID_MODEL_LANGUAGES:
    raise ValueError(
      f"Language '{lang}' is not supported by the model. Available languages are: {', '.join(VALID_MODEL_LANGUAGES)}."
    )


def _check_is_valid_species_list_path(species_list: Path) -> None:
  if not species_list.is_file():
    raise ValueError(f"Species list file '{species_list.absolute()}' does not exist!")


def _check_is_valid_pb_model_folder(folder_path: Path) -> None:
  if not folder_path.is_dir():
    raise ValueError(f"Model folder '{folder_path.absolute()}' does not exist!")
  if not check_protobuf_model_files_exist(folder_path):
    raise ValueError(
      f"Model folder '{folder_path.absolute()}' does not contain valid protobuf model files!"
    )


def _check_is_valid_tf_file(model: str | PathLike[str]) -> None:
  model_path = Path(model)
  if not model_path.is_file():
    raise ValueError(f"Model file '{model_path.absolute()}' does not exist!")
  if not model_path.suffix == ".tflite":
    raise ValueError(
      f"Model file '{model_path.absolute()}' is not a valid TFLite model file!"
    )


def load(
  model_type: Literal["acoustic", "geo"],
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  lang: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US,
) -> ModelBase:
  _check_is_valid_model_type(model_type)

  if model_type == "acoustic":
    return _load_acoustic_model(
      version=cast(ACOUSTIC_MODEL_VERSIONS, version),
      backend=backend,
      precision=precision,
      lang=lang,
    )
  elif model_type == "geo":
    if precision != MODEL_PRECISION_FLOAT32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
      )
    return _load_geo_model(
      version=cast(GEO_MODEL_VERSIONS, version),
      backend=backend,
      lang=lang,
    )
  else:
    raise AssertionError()


def _load_acoustic_model(
  version: ACOUSTIC_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
) -> AcousticModelBase:
  _check_is_valid_acoustic_model_version(version)

  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    return _load_acoustic_model_V2_4(
      backend=backend,
      precision=precision,
      lang=lang,
    )
  else:
    raise AssertionError()


def _load_acoustic_model_V2_4(
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
) -> AcousticModelBaseV2_4:
  _check_is_valid_backend(backend)
  _check_is_valid_precision(precision)
  _check_is_valid_language(lang)

  if backend == MODEL_BACKEND_TF:
    return AcousticTFModelV2_4.load_official(lang, precision)
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FLOAT32:
      raise ValueError(
        f"Unsupported model precision for acoustic pb model: {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
      )

    return AcousticPBModelV2_4.load_official(lang)
  else:
    raise AssertionError()


def _load_geo_model(
  version: GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  lang: MODEL_LANGUAGES,
) -> GeoModelBase:
  _check_is_valid_geo_model_version(version)

  if version == GEO_MODEL_VERSION_V2_4:
    return _load_geo_model_V2_4(backend, lang)
  else:
    raise AssertionError()


def _load_geo_model_V2_4(
  backend: MODEL_BACKENDS,
  lang: MODEL_LANGUAGES,
) -> GeoModelBase:
  _check_is_valid_backend(backend)
  _check_is_valid_language(lang)

  if backend == MODEL_BACKEND_TF:
    return GeoTFModelV2_4.load_official(lang)
  elif backend == MODEL_BACKEND_PB:
    return GeoPBModelV2_4.load_official(lang)
  else:
    raise AssertionError()


def load_custom(
  model_type: Literal["acoustic", "geo"],
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  model: str | PathLike[str],
  species_list: str | PathLike[str],
  /,
  *,
  precision: MODEL_PRECISIONS = MODEL_PRECISION_FLOAT32,
  check_validity: bool = True,
) -> ModelBase:
  _check_is_valid_model_type(model_type)

  if model_type == "acoustic":
    return _load_custom_acoustic_model(
      version=cast(ACOUSTIC_MODEL_VERSIONS, version),
      backend=backend,
      precision=precision,
      model=Path(model),
      species_list=Path(species_list),
      check_validity=check_validity,
    )
  elif model_type == "geo":
    if precision != MODEL_PRECISION_FLOAT32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
      )
    return _load_custom_geo_model(
      version=cast(GEO_MODEL_VERSIONS, version),
      backend=backend,
      model=Path(model),
      species_list=Path(species_list),
      check_validity=check_validity,
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
) -> AcousticModelBase:
  _check_is_valid_acoustic_model_version(version)

  if version == ACOUSTIC_MODEL_VERSION_V2_4:
    return _load_custom_acoustic_model_V2_4(
      backend=backend,
      precision=precision,
      species_list=species_list,
      model=model,
      check_validity=check_validity,
    )
  else:
    raise AssertionError()


def _load_custom_acoustic_model_V2_4(
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  model: Path,
  species_list: Path,
  check_validity: bool,
) -> AcousticModelBaseV2_4:
  model = Path(model)
  species_list = Path(species_list)

  _check_is_valid_backend(backend)
  _check_is_valid_precision(precision)
  _check_is_valid_species_list_path(species_list)

  if backend == MODEL_BACKEND_TF:
    _check_is_valid_tf_file(model)
    return AcousticTFModelV2_4.load_custom(
      model, species_list, precision, check_validity
    )
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FLOAT32:
      raise ValueError(
        f"Unsupported model precision for acoustic pb model: {precision}. Currently supported precision is: {MODEL_PRECISION_FLOAT32}."
      )
    _check_is_valid_pb_model_folder(model)
    return AcousticPBModelV2_4.load_custom(model, species_list, check_validity)
  else:
    raise AssertionError()


def _load_custom_geo_model(
  version: GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  model: Path,
  species_list: Path,
  check_validity: bool,
) -> GeoModelBase:
  _check_is_valid_geo_model_version(version)

  if version == GEO_MODEL_VERSION_V2_4:
    return _load_custom_geo_model_V2_4(backend, model, species_list, check_validity)
  else:
    raise AssertionError()


def _load_custom_geo_model_V2_4(
  backend: MODEL_BACKENDS,
  model: Path,
  species_list: Path,
  check_validity: bool,
) -> GeoModelBase:
  _check_is_valid_backend(backend)
  _check_is_valid_species_list_path(species_list)

  if backend == MODEL_BACKEND_TF:
    _check_is_valid_tf_file(model)
    return GeoTFModelV2_4.load_custom(model, species_list, check_validity)
  elif backend == MODEL_BACKEND_PB:
    _check_is_valid_pb_model_folder(model)
    return GeoPBModelV2_4.load_custom(model, species_list, check_validity)
  else:
    raise AssertionError()
