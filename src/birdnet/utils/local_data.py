import functools
import importlib.metadata
import os
from pathlib import Path

from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  ENV_VAR_APP_DATA,
  GEO_MODEL_VERSIONS,
  MODEL_BACKEND_ONNX,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_PT,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
  MODEL_TYPES,
  PKG_NAME,
)


def get_package_version() -> str:
  return importlib.metadata.version(PKG_NAME)


def get_app_data_path() -> Path:
  app_data_path: str
  if os.name == "nt":  # Windows
    path = os.getenv("APPDATA")
    assert path is not None
    app_data_path = path
  elif os.name == "posix":
    if os.uname().sysname == "Darwin":  # Mac OS X
      app_data_path = os.path.expanduser("~/Library/Application Support")
    else:  # Linux
      app_data_path = os.path.expanduser("~/.local/share")
  else:
    raise OSError("Unsupported operating system")

  result = Path(app_data_path)
  return result


@functools.cache
def _resolve_override(raw: str) -> Path:
  # Once per value: a relative override must not move with a later chdir.
  return Path(raw).expanduser().resolve()


def get_birdnet_app_data_folder() -> Path:
  override = os.getenv(ENV_VAR_APP_DATA)
  if override is not None:
    return _resolve_override(override)
  app_data = get_app_data_path()
  result = app_data / PKG_NAME
  return result


# Import-time value for downstream imports; birdnet calls get_birdnet_app_data_folder().
APP_DIR = get_birdnet_app_data_folder()


def get_benchmark_dir(
  model: MODEL_TYPES,
  dir_name: str,
) -> Path:
  result = (
    get_birdnet_app_data_folder()
    / f"{model}-benchmarks"
    / f"lib-v{get_package_version()}"
    / f"{dir_name}"
  )
  result.mkdir(parents=True, exist_ok=True)
  return result


def get_model_root_dir(
  model: MODEL_TYPES,
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
) -> Path:
  parent_dir = (
    get_birdnet_app_data_folder() / f"{model}-models" / f"v{version}" / backend
  )
  return parent_dir


def get_model_path(
  model: MODEL_TYPES,
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
  content_tag: str | None = None,
) -> Path:
  # `version` names the model family (a "v3.0" preview or geomodel release keeps
  # its directory across retrains), so for the checksum-verified downloads the
  # content tag is what tells one release's file from another's.
  stem = f"model-{precision}"
  if content_tag is not None:
    stem = f"{stem}-{content_tag}"
  root_dir = get_model_root_dir(model, version, backend)
  if backend == MODEL_BACKEND_TF:
    result = root_dir / f"{stem}.tflite"
  elif backend == MODEL_BACKEND_PB:
    result = root_dir / stem
  elif backend == MODEL_BACKEND_PT:
    result = root_dir / f"{stem}.pt"
  elif backend == MODEL_BACKEND_ONNX:
    result = root_dir / f"{stem}.onnx"
  else:
    raise AssertionError()
  return result


def get_lang_dir(
  model: MODEL_TYPES,
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
) -> Path:
  root_dir = get_model_root_dir(model, version, backend)
  result = root_dir / "labels"
  return result
