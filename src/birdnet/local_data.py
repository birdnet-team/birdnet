import importlib.metadata
import os
import tempfile
from pathlib import Path

from birdnet.base import MODEL_BACKENDS, MODEL_TYPES, MODEL_VERSIONS
from birdnet.globals import PKG_NAME


def get_package_version() -> str:
  return importlib.metadata.version(PKG_NAME)


def get_app_data_path() -> Path:
  app_data_path: str
  if os.name == "nt":  # Windows
    app_data_path = os.getenv("APPDATA")
    assert app_data_path is not None
  elif os.name == "posix":
    if os.uname().sysname == "Darwin":  # Mac OS X
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
  version: MODEL_VERSIONS,
) -> Path:
  result = (
    # Path(tempfile.gettempdir()) / f"{PKG_NAME}-benchmarks" / f"{model}-v{version}"
    APP_DIR / f"{model}-benchmarks" / f"v{version}" / f"lib-v{get_package_version()}"
  )
  result.mkdir(parents=True, exist_ok=True)
  return result


def get_local_model_root_dir(
  model: MODEL_TYPES,
  version: MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
) -> Path:
  parent_dir = APP_DIR / f"{model}-models" / f"v{version}" / backend
  return parent_dir


if not APP_DIR.exists():
  APP_DIR.mkdir(parents=True, exist_ok=True)
