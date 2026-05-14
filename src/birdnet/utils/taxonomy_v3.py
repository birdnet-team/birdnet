from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from birdnet.utils.helper import download_file_tqdm
from birdnet.utils.local_data import APP_DIR

_TAXONOMY_V3_DL_URL = (
  "https://github.com/birdnet-team/geomodel/raw/refs/tags/v3.0.2/taxonomy.csv"
)
_TAXONOMY_V3_DL_SIZE = 9162669
_TAXONOMY_V3_PATH = APP_DIR / "taxonomy_v3_0.csv"
_TAXONOMY_V3_LOCK_DIR = APP_DIR / ".taxonomy_v3_0.lock"


def get_taxonomy_v3_path() -> Path:
  return _TAXONOMY_V3_PATH


def taxonomy_v3_available() -> bool:
  if not _TAXONOMY_V3_PATH.is_file():
    return False
  return _TAXONOMY_V3_PATH.stat().st_size == _TAXONOMY_V3_DL_SIZE


@contextmanager
def _taxonomy_v3_lock(timeout_s: float = 300.0) -> Generator[None, None, None]:
  deadline = time.monotonic() + timeout_s
  while True:
    try:
      _TAXONOMY_V3_LOCK_DIR.mkdir(parents=True, exist_ok=False)
      break
    except FileExistsError as err:
      if time.monotonic() >= deadline:
        raise TimeoutError(
          "Timed out while waiting for the shared v3.0 taxonomy setup."
        ) from err
      time.sleep(0.1)

  try:
    yield
  finally:
    _TAXONOMY_V3_LOCK_DIR.rmdir()


def ensure_taxonomy_v3_available() -> Path:
  with _taxonomy_v3_lock():
    if taxonomy_v3_available():
      return _TAXONOMY_V3_PATH

    _TAXONOMY_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    download_file_tqdm(
      _TAXONOMY_V3_DL_URL,
      _TAXONOMY_V3_PATH,
      download_size=_TAXONOMY_V3_DL_SIZE,
      description="Downloading shared v3.0 taxonomy",
    )

  return _TAXONOMY_V3_PATH
