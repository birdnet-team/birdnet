from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic_models.perch_v2.pb import (
  AcousticPBBackendFP32PerchV2,
  AcousticPBDownloaderPerchV2,
)
from birdnet.acoustic_models.v2_4.model import (
  AcousticModelV2_4,
)
from birdnet.acoustic_models.v2_4.pb import (
  AcousticPBBackendFP32V2_4,
  AcousticPBDownloaderV2_4,
  AcousticRavenBackendFP32V2_4,
)
from birdnet.acoustic_models.v2_4.tf import (
  AcousticTFBackendFP16V2_4,
  AcousticTFBackendFP32V2_4,
  AcousticTFBackendInt8V2_4,
  AcousticTFDownloaderV2_4,
)
from birdnet.backends import (
  TF_BACKEND_LIB_ARG,
  VersionedAcousticBackendProtocol,
  litert_installed,
  tf_installed,
)
from birdnet.base import ModelBase
from birdnet.geo_models.base import GeoModelBase
from birdnet.geo_models.v2_4.model import (
  GeoModelV2_4,
)
from birdnet.geo_models.v2_4.pb import GeoPBBackendFP32V2_4, GeoPBDownloaderV2_4
from birdnet.geo_models.v2_4.tf import (
  GeoTFBackendFP32V2_4,
  GeoTFDownloaderV2_4,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  CUSTOM_PB_IS_RAVEN_DEFAULT,
  CUSTOM_PB_IS_RAVEN_PARAM,
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  LIBRARY_LITERT,
  LIBRARY_TF_DEFAULT,
  LIBRARY_TF_PARAM,
  LIBRARY_TFLITE,
  LIBRARY_TYPES,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGE_EN_US,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
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


def _validate_model_type(model_type: Any) -> MODEL_TYPES:  # noqa: ANN401
  if model_type not in VALID_MODEL_TYPES:
    raise ValueError(
      f"Unknown model type: {model_type}. "
      f"Supported types are: {', '.join(VALID_MODEL_TYPES)}."
    )
  return cast(MODEL_TYPES, model_type)


def _validate_acoustic_model_version(version: Any) -> ACOUSTIC_MODEL_VERSIONS:  # noqa: ANN401
  if version not in VALID_ACOUSTIC_MODEL_VERSIONS:
    raise ValueError(
      f"Unsupported model version: {version}. "
      f"Available versions are: {', '.join(VALID_ACOUSTIC_MODEL_VERSIONS)}."
    )
  return cast(ACOUSTIC_MODEL_VERSIONS, version)


def _validate_geo_model_version(version: Any) -> GEO_MODEL_VERSIONS:  # noqa: ANN401
  if version not in VALID_GEO_MODEL_VERSIONS:
    raise ValueError(
      f"Unsupported model version: {version}. "
      f"Available versions are: {', '.join(VALID_GEO_MODEL_VERSIONS)}."
    )
  return cast(GEO_MODEL_VERSIONS, version)


def _validate_backend(backend: Any) -> MODEL_BACKENDS:  # noqa: ANN401
  if backend not in VALID_MODEL_BACKENDS:
    raise ValueError(
      f"Unknown model backend: {backend}. "
      f"Available backends are: {', '.join(VALID_MODEL_BACKENDS)}."
    )
  return cast(MODEL_BACKENDS, backend)


def _validate_precision(precision: Any) -> MODEL_PRECISIONS:  # noqa: ANN401
  if precision not in VALID_MODEL_PRECISIONS:
    raise ValueError(
      f"Unsupported model precision: {precision}. "
      f"Currently supported precisions: {', '.join(VALID_MODEL_PRECISIONS)}."
    )
  return cast(MODEL_PRECISIONS, precision)


def _validate_language(lang: Any) -> MODEL_LANGUAGES:  # noqa: ANN401
  if lang not in VALID_MODEL_LANGUAGES:
    raise ValueError(
      f"Language '{lang}' is not supported by the model. "
      f"Available languages are: {', '.join(VALID_MODEL_LANGUAGES)}."
    )
  return cast(MODEL_LANGUAGES, lang)


def _validate_species_list_path(species_list: Any | PathLike[Any]) -> Path:  # noqa: ANN401
  species_list = Path(species_list)
  if not species_list.is_file():
    raise ValueError(f"Species list file '{species_list.absolute()}' does not exist!")
  return species_list


def _validate_path(path: Any) -> Path:  # noqa: ANN401
  path = Path(path)
  if not path.exists():
    raise ValueError(f"Path '{path.absolute()}' does not exist!")
  return path


def _validate_pb_model_folder(folder_path: Any) -> Path:  # noqa: ANN401
  path = Path(folder_path)
  if not path.is_dir():
    raise ValueError(f"Model folder '{path.absolute()}' does not exist!")
  if not check_protobuf_model_files_exist(path):
    raise ValueError(
      f"Model folder '{path.absolute()}' does not contain valid protobuf model files!"
    )
  return path


def _validate_tf_file(model_path: Any) -> Path:  # noqa: ANN401
  path = Path(model_path)
  if not path.is_file():
    raise ValueError(f"Model file '{path.absolute()}' does not exist!")
  if not path.suffix == ".tflite":
    raise ValueError(
      f"Model file '{path.absolute()}' is not a valid TFLite model file!"
    )
  return path


def _validate_library(library: Any) -> LIBRARY_TYPES:  # noqa: ANN401
  if library not in VALID_LIBRARY_TYPES:
    raise ValueError(
      f"Unsupported TensorFlow library: {library}. "
      f"Supported libraries are: {', '.join(VALID_LIBRARY_TYPES)}."
    )
  if library == LIBRARY_TFLITE:
    assert tf_installed()  # default
  elif library == LIBRARY_LITERT:
    if not litert_installed():
      raise ValueError(
        f"Parameter 'library': Library '{LIBRARY_LITERT}' is not available. "
        "Install birdnet with [litert] option."
      )
  else:
    raise AssertionError()
  return cast(LIBRARY_TYPES, library)


def _validate_custom_pb_is_raven(is_raven: Any) -> bool:  # noqa: ANN401
  if not isinstance(is_raven, bool):
    raise ValueError(
      f"Parameter '{CUSTOM_PB_IS_RAVEN_PARAM}' must be of type bool, "
      f"got {type(is_raven)}."
    )
  return is_raven


def _validate_kwargs_allowed(
  model_kwargs: dict, allowed: set[str] | None
) -> dict[str, Any]:
  if allowed is None:
    not_allowed = set(model_kwargs.keys())
  else:
    not_allowed = set(model_kwargs.keys()) - allowed
  if len(not_allowed) > 0:
    raise ValueError(f"Unexpected keyword arguments: {', '.join(not_allowed)}. ")

  if allowed is None:
    model_kwargs = {}

  return model_kwargs


def _validate_device(device: Any) -> Literal["CPU", "GPU"]:  # noqa: ANN401
  if device not in ("CPU", "GPU"):
    raise ValueError(f"Unknown device: {device}. Supported devices are: CPU, GPU.")
  return cast(Literal["CPU", "GPU"], device)


def load_perch_v2(device: str) -> AcousticModelPerchV2:
  device = _validate_device(device)
  model_path, species_list = AcousticPBDownloaderPerchV2.get_model_path_and_labels(
    device
  )

  backend_type: type[VersionedAcousticBackendProtocol]
  backend_type = AcousticPBBackendFP32PerchV2

  return AcousticModelPerchV2.load(
    model_path,
    species_list,
    backend_type=backend_type,
    backend_kwargs={},
  )


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
  model_type = _validate_model_type(model_type)
  backend = _validate_backend(backend)
  precision = _validate_precision(precision)
  lang = _validate_language(lang)

  if model_type == MODEL_TYPE_ACOUSTIC:
    version = _validate_acoustic_model_version(version)
    return _load_acoustic_model(
      version=version,
      backend=backend,
      precision=precision,
      lang=lang,
      **model_kwargs,
    )
  elif model_type == MODEL_TYPE_GEO:
    version = _validate_geo_model_version(version)
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
    return _load_geo_model_V2_4(backend, precision, lang, **model_kwargs)
  else:
    raise AssertionError()


def _load_acoustic_model_V2_4(
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
  **model_kwargs: object,
) -> AcousticModelV2_4:
  if backend == MODEL_BACKEND_TF:
    model_kwargs = _validate_kwargs_allowed(model_kwargs, {LIBRARY_TF_PARAM})
    library = _validate_library(model_kwargs.get(LIBRARY_TF_PARAM, LIBRARY_TF_DEFAULT))

    model_path, species_list = AcousticTFDownloaderV2_4.get_model_path_and_labels(
      lang, precision
    )

    backend_type: type[VersionedAcousticBackendProtocol]
    if precision == MODEL_PRECISION_INT8:
      backend_type = AcousticTFBackendInt8V2_4
    elif precision == MODEL_PRECISION_FP16:
      backend_type = AcousticTFBackendFP16V2_4
    else:
      assert precision == MODEL_PRECISION_FP32
      backend_type = AcousticTFBackendFP32V2_4

    return AcousticModelV2_4.load(
      model_path,
      species_list,
      backend_type=backend_type,
      backend_kwargs={
        TF_BACKEND_LIB_ARG: library,
      },
    )
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for acoustic pb model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )
    model_kwargs = _validate_kwargs_allowed(model_kwargs, None)

    model_path, species_list = AcousticPBDownloaderV2_4.get_model_path_and_labels(lang)
    return AcousticModelV2_4.load(
      model_path,
      species_list,
      backend_type=AcousticPBBackendFP32V2_4,
      backend_kwargs={},
    )
  else:
    raise AssertionError()


def _load_geo_model_V2_4(
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  lang: MODEL_LANGUAGES,
  **model_kwargs: object,
) -> GeoModelV2_4:
  if backend == MODEL_BACKEND_TF:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )

    model_kwargs = _validate_kwargs_allowed(model_kwargs, {LIBRARY_TF_PARAM})
    library = _validate_library(model_kwargs.get(LIBRARY_TF_PARAM, LIBRARY_TF_DEFAULT))

    model_path, species_list = GeoTFDownloaderV2_4.get_model_path_and_labels(lang)

    return GeoModelV2_4.load(
      model_path,
      species_list,
      backend_type=GeoTFBackendFP32V2_4,
      backend_kwargs={
        TF_BACKEND_LIB_ARG: library,
      },
    )
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )

    model_kwargs = _validate_kwargs_allowed(model_kwargs, None)

    model_path, species_list = GeoPBDownloaderV2_4.get_model_path_and_labels(lang)
    return GeoModelV2_4.load(
      model_path,
      species_list,
      backend_type=GeoPBBackendFP32V2_4,
      backend_kwargs={},
    )
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
  model_type = _validate_model_type(model_type)
  backend = _validate_backend(backend)
  model = _validate_path(model)
  species_list = _validate_species_list_path(species_list)
  precision = _validate_precision(precision)

  if model_type == MODEL_TYPE_ACOUSTIC:
    version = _validate_acoustic_model_version(version)
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
    version = _validate_geo_model_version(version)
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
    return _load_custom_geo_model_V2_4(
      backend, model, precision, species_list, check_validity, **model_kwargs
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
) -> AcousticModelV2_4:
  if backend == MODEL_BACKEND_TF:
    model = _validate_tf_file(model)
    model_kwargs = _validate_kwargs_allowed(model_kwargs, {LIBRARY_TF_PARAM})
    library = _validate_library(model_kwargs.get(LIBRARY_TF_PARAM, LIBRARY_TF_DEFAULT))

    backend_type: type[VersionedAcousticBackendProtocol]
    if precision == MODEL_PRECISION_INT8:
      backend_type = AcousticTFBackendInt8V2_4
    elif precision == MODEL_PRECISION_FP16:
      backend_type = AcousticTFBackendFP16V2_4
    else:
      assert precision == MODEL_PRECISION_FP32
      backend_type = AcousticTFBackendFP32V2_4

    return AcousticModelV2_4.load_custom(
      model,
      species_list,
      backend_type=backend_type,
      backend_kwargs={
        TF_BACKEND_LIB_ARG: library,
      },
      check_validity=check_validity,
    )
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for acoustic pb model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )

    model = _validate_pb_model_folder(model)
    model_kwargs = _validate_kwargs_allowed(model_kwargs, {CUSTOM_PB_IS_RAVEN_PARAM})
    is_raven = _validate_custom_pb_is_raven(
      model_kwargs.get(CUSTOM_PB_IS_RAVEN_PARAM, CUSTOM_PB_IS_RAVEN_DEFAULT)
    )

    backend_type: type[VersionedAcousticBackendProtocol]
    if is_raven:
      backend_type = AcousticRavenBackendFP32V2_4
    else:
      backend_type = AcousticPBBackendFP32V2_4

    return AcousticModelV2_4.load_custom(
      model,
      species_list,
      backend_type=backend_type,
      backend_kwargs={},
      check_validity=check_validity,
    )
  else:
    raise AssertionError()


def _load_custom_geo_model_V2_4(
  backend: MODEL_BACKENDS,
  model: Path,
  precision: MODEL_PRECISIONS,
  species_list: Path,
  check_validity: bool,
  **model_kwargs: object,
) -> GeoModelV2_4:
  if backend == MODEL_BACKEND_TF:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )

    model = _validate_tf_file(model)
    model_kwargs = _validate_kwargs_allowed(model_kwargs, {LIBRARY_TF_PARAM})
    library = _validate_library(model_kwargs.get(LIBRARY_TF_PARAM, LIBRARY_TF_DEFAULT))

    return GeoModelV2_4.load_custom(
      model,
      species_list,
      backend_type=GeoTFBackendFP32V2_4,
      backend_kwargs={
        TF_BACKEND_LIB_ARG: library,
      },
      check_validity=check_validity,
    )
  elif backend == MODEL_BACKEND_PB:
    if precision != MODEL_PRECISION_FP32:
      raise ValueError(
        f"Unsupported model precision for geo model: {precision}. "
        f"Currently supported precision is: {MODEL_PRECISION_FP32}."
      )

    model = _validate_pb_model_folder(model)
    model_kwargs = _validate_kwargs_allowed(model_kwargs, None)

    return GeoModelV2_4.load_custom(
      model,
      species_list,
      backend_type=GeoPBBackendFP32V2_4,
      backend_kwargs={},
      check_validity=check_validity,
    )
  else:
    raise AssertionError()
