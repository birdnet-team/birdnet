from pathlib import Path

from birdnet_v2.base import MODEL_BACKENDS, MODEL_TYPES, MODEL_VERSIONS
from birdnet_v2.globals import APP_DIR


def get_local_model_root_dir(
  model: MODEL_TYPES,
  version: MODEL_VERSIONS,
  backend: MODEL_BACKENDS,
) -> Path:
  parent_dir = APP_DIR / f"{model}-models" / f"v{version}" / backend
  return parent_dir
