"""The taxonomy shared by the V3.0 models, and the localized labels built from it.

The taxonomy is a single CSV published by the geomodel repository (versioned
since its v3.0.4 release) and cached under a generic name in the app data
directory. Both V3.0 models use it; the V2.4 models ship static label files and
do not.

What it is *not*: it never decides what a model predicts. Species identity, count
and order come from that model's own label file. The taxonomy only supplies the
common name per language, and a species it cannot resolve falls back to the
English name from the label file - silently, by design.

The two models join to it on **different keys**, deliberately:

- geo (``geo/models/v3_0/model.py``) joins on ``species_code``. Its label file
  ships one and all of them resolve, and codes survive the taxonomy updates that
  rename species. Where upstream reuses one code for two species, the row whose
  ``sci_name`` matches the label wins.
- acoustic (``acoustic/models/v3_0/model.py``) joins on ``sci_name``, because its
  label file carries no code the taxonomy shares (its ``id`` column matches
  nothing in it). That is why it leaves a few hundred species unresolved where
  geo leaves none: scientific names drift between taxonomy versions.

The two live in separate files, so a change to one is worth checking against the
other.

Every cached artifact has a generic on-disk name, so staleness is inferred:

- the taxonomy CSV and each raw label file: exact byte size vs. the constants
  here and in the model modules;
- the generated ``<lang>.txt`` files: one line per raw label line, plus the
  ``.birdnet_taxonomy`` marker written beside them. The marker is what catches a
  taxonomy-only bump - the taxonomy being shared means the first model to
  download a new one makes it "available" for all the others, whose label files
  would otherwise look current while still holding the previous release's names.

Bumping the taxonomy is therefore: update the URL and size below, then check that
every column in both models' ``_LANGUAGE_TO_COLUMN`` still exists in the new file.
A missing column does not raise - it yields a complete file of English names,
which is how Estonian outlived the column being dropped.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from birdnet.utils.helper import download_file_tqdm
from birdnet.utils.local_data import APP_DIR

# The geomodel repository versions its taxonomy since v3.0.4 (the unversioned
# taxonomy.csv is gone). This file is the one the geo model v3.0.4 was built
# against: all of its 14,082 label codes resolve in it. The acoustic model v3.0
# shares it (matching by scientific name), where it is also the closer fit - it
# resolves 188 more of the acoustic species than the previous v3.0.2 taxonomy,
# and its English names agree with the acoustic label file in 1,487 cases where
# the old one disagreed.
_TAXONOMY_V3_DL_URL = "https://github.com/birdnet-team/geomodel/raw/refs/tags/v3.0.4/taxonomy_v0.2-Jun2026.csv"
_TAXONOMY_V3_DL_SIZE = 11078402
_TAXONOMY_V3_PATH = APP_DIR / "taxonomy_v3_0.csv"
_TAXONOMY_V3_LOCK_DIR = APP_DIR / ".taxonomy_v3_0.lock"
# Written into a generated label directory to record which taxonomy the localized
# names came from. The taxonomy is shared between the v3.0 models and its on-disk
# name is generic, so without this a label directory generated from an older
# taxonomy is indistinguishable from a current one: whichever model downloads the
# new taxonomy first makes it "available" for all the others, and they then keep
# serving names generated from the previous one.
_TAXONOMY_MARKER_NAME = ".birdnet_taxonomy"


def get_taxonomy_v3_path() -> Path:
  return _TAXONOMY_V3_PATH


def taxonomy_v3_available() -> bool:
  if not _TAXONOMY_V3_PATH.is_file():
    return False
  return _TAXONOMY_V3_PATH.stat().st_size == _TAXONOMY_V3_DL_SIZE


def taxonomy_v3_marker_matches(lang_dir: Path) -> bool:
  """Whether `lang_dir` was generated from the taxonomy that is current now."""
  marker = lang_dir / _TAXONOMY_MARKER_NAME
  if not marker.is_file():
    return False
  return marker.read_text(encoding="utf-8").strip() == _TAXONOMY_V3_DL_URL


def write_taxonomy_v3_marker(lang_dir: Path) -> None:
  (lang_dir / _TAXONOMY_MARKER_NAME).write_text(_TAXONOMY_V3_DL_URL, encoding="utf-8")


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
    with suppress(FileNotFoundError):
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
