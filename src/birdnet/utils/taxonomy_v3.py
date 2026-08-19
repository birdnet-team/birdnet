"""The taxonomy shared by the V3.0 models, and the localized labels built from it.

The taxonomy is a single CSV published by the geomodel repository (versioned
since its v3.0.4 release) and cached in the app data directory under the file
name from its URL. Both V3.0 models use it; the V2.4 models ship static label
files and do not.

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

Staleness is decided from what a directory records about itself, not from what
the running version expects to find. Each generated label directory carries a
``.birdnet_labels.json`` manifest holding the digests of the inputs that were
actually read and of every file that was written; see
``birdnet/utils/label_manifest.py`` for why digests and not names.

The taxonomy is cached under the file name from its URL rather than a generic
one, so two releases cannot occupy the same path. That matters because the
taxonomy is shared: under a generic name, a second installed version judges this
release stale by byte size, downloads its own over the top, and leaves every
other version reading a taxonomy it never asked for.

Bumping the taxonomy is therefore: update the URL, size and SHA-256 below, then
check that every column in both models' ``_LANGUAGE_TO_COLUMN`` still exists in
the new file. A missing column does not raise - it yields a complete file of
English names, which is how Estonian outlived the column being dropped. Changing
either model's generation logic means bumping its ``_GENERATION_VERSION``; the
golden-digest tests fail until both that and the expected digests are updated.
"""

from __future__ import annotations

import os
from pathlib import Path

from birdnet.utils.helper import directory_lock, download_file_tqdm
from birdnet.utils.label_manifest import LabelInput, sha256_file
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
_TAXONOMY_V3_DL_SHA256 = (
  "98b27fc4a77c5e321c7bbf96f924fc4b58170de9688e79ebf3ea8263d522580a"
)

# The file name comes from the URL, so two releases cannot occupy one path. Under
# the previous generic name a second installed version would judge this release
# stale by byte size, download its own over the top, and leave every other
# version reading a taxonomy it never asked for.
_TAXONOMY_V3_DIR = APP_DIR / "taxonomy-v3"
_TAXONOMY_V3_PATH = _TAXONOMY_V3_DIR / _TAXONOMY_V3_DL_URL.rsplit("/", 1)[-1]
_LEGACY_TAXONOMY_V3_PATH = APP_DIR / "taxonomy_v3_0.csv"
_TAXONOMY_V3_LOCK_DIR = APP_DIR / ".taxonomy_v3_0.lock"


def get_taxonomy_v3_path() -> Path:
  return _TAXONOMY_V3_PATH


def get_taxonomy_v3_input() -> LabelInput:
  """The taxonomy as an input a generated label directory can record."""
  return LabelInput(
    path=_TAXONOMY_V3_PATH,
    url=_TAXONOMY_V3_DL_URL,
    size=_TAXONOMY_V3_DL_SIZE,
    sha256=_TAXONOMY_V3_DL_SHA256,
  )


def taxonomy_v3_available() -> bool:
  if not _TAXONOMY_V3_PATH.is_file():
    return False
  return _TAXONOMY_V3_PATH.stat().st_size == _TAXONOMY_V3_DL_SIZE


def _adopt_legacy_taxonomy() -> bool:
  """Move a correct taxonomy from the pre-release path instead of downloading it.

  Keeps the upgrade offline for everyone already holding this release. A file
  that hashes differently is left where it is: it belongs to another version
  that is still reading it from there.
  """
  if _TAXONOMY_V3_PATH.is_file() or not _LEGACY_TAXONOMY_V3_PATH.is_file():
    return False
  if _LEGACY_TAXONOMY_V3_PATH.stat().st_size != _TAXONOMY_V3_DL_SIZE:
    return False
  if sha256_file(_LEGACY_TAXONOMY_V3_PATH) != _TAXONOMY_V3_DL_SHA256:
    return False
  _TAXONOMY_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
  os.replace(_LEGACY_TAXONOMY_V3_PATH, _TAXONOMY_V3_PATH)
  return True


def ensure_taxonomy_v3_available() -> Path:
  with directory_lock(_TAXONOMY_V3_LOCK_DIR, "the shared v3.0 taxonomy setup"):
    if taxonomy_v3_available():
      return _TAXONOMY_V3_PATH

    if _adopt_legacy_taxonomy():
      return _TAXONOMY_V3_PATH

    _TAXONOMY_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    download_file_tqdm(
      _TAXONOMY_V3_DL_URL,
      _TAXONOMY_V3_PATH,
      download_size=_TAXONOMY_V3_DL_SIZE,
      description="Downloading shared v3.0 taxonomy",
    )
    actual = sha256_file(_TAXONOMY_V3_PATH)
    if actual != _TAXONOMY_V3_DL_SHA256:
      _TAXONOMY_V3_PATH.unlink(missing_ok=True)
      raise RuntimeError(
        f"The shared v3.0 taxonomy downloaded from {_TAXONOMY_V3_DL_URL} does "
        f"not match its expected checksum ({actual} instead of "
        f"{_TAXONOMY_V3_DL_SHA256}). The file was discarded; retry, and if this "
        "persists the published file has changed."
      )

  return _TAXONOMY_V3_PATH
