import importlib.metadata
import os
from pathlib import Path

from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  GEO_MODEL_VERSIONS,
  MODEL_BACKEND_PB,
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
      # e.g., /Users/runner/Library/Application Support
      app_data_path = os.path.expanduser("~/Library/Application Support")
    else:  # Linux
      app_data_path = os.path.expanduser("~/.local/share")
  else:
    raise OSError("Unsupported operating system")

  result = Path(app_data_path)
  return result


def get_birdnet_app_data_folder() -> Path:
  app_data = get_app_data_path()
  result = app_data / PKG_NAME
  return result


APP_DIR = get_birdnet_app_data_folder()


def get_benchmark_dir(
  model: MODEL_TYPES,
  dir_name: str,
) -> Path:
  result = (
    APP_DIR / f"{model}-benchmarks" / f"lib-v{get_package_version()}" / f"{dir_name}"
  )
  result.mkdir(parents=True, exist_ok=True)
  return result


def get_model_root_dir(
  model: MODEL_TYPES,
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
) -> Path:
  parent_dir = APP_DIR / f"{model}-models" / f"v{version}" / backend
  return parent_dir


def get_model_path(
  model: MODEL_TYPES,
  version: ACOUSTIC_MODEL_VERSIONS | GEO_MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
  precision: MODEL_PRECISIONS,
) -> Path:
  root_dir = get_model_root_dir(model, version, backend)
  if backend == MODEL_BACKEND_TF:
    result = root_dir / f"model-{precision}.tflite"
  elif backend == MODEL_BACKEND_PB:
    result = root_dir / f"model-{precision}"
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


if not APP_DIR.exists():
  APP_DIR.mkdir(parents=True, exist_ok=True)
