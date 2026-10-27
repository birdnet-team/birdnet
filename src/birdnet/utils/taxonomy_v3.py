"""The taxonomy shared by the V3.0 models, and the localized labels built from it.

The taxonomy is a single CSV published by the geomodel repository (versioned
since its v3.0.4 release) and cached in the app data directory, in a directory
named after its SHA-256. Both V3.0 models use it; the V2.4 models ship static
label files and do not.

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

Each model's label parsing and join live in its own file, so a change to one is
worth checking against the other. The rest of the generation is shared, in
``birdnet/utils/v3_labels.py``.

Staleness is decided from what a directory records about itself, not from what
the running version expects to find. Each generated label directory carries a
``.birdnet_labels.json`` manifest holding the digests of the inputs that were
actually read and of every file that was written; see
``birdnet/utils/label_manifest.py`` for why digests and not names.

Two releases never share a cache path; see ``_taxonomy_v3_dir``.

Bumping the taxonomy is therefore: update the URL, size and SHA-256 below, then
check that every column in both models' ``_LANGUAGE_TO_COLUMN`` still exists in
the new file. A missing column does not raise - it yields a complete file of
English names, which is how Estonian outlived the column being dropped. Changing
either model's generation logic means bumping its ``_GENERATION_VERSION``, and
changing the shared part in ``v3_labels.py`` means bumping both. The
golden-digest tests fail when the bytes generated from their fixtures move;
nothing forces the bump when their digests are updated.
"""

from __future__ import annotations

from pathlib import Path

from birdnet.utils.filesystem import directory_lock
from birdnet.utils.label_manifest import (
  LabelInput,
  artifact_is_current,
  ensure_artifact,
)
from birdnet.utils.local_data import get_birdnet_app_data_folder

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


# Named after the content, so two releases cannot occupy one path, not even if
# upstream re-publishes under the same file name.
def _taxonomy_v3_dir() -> Path:
  return get_birdnet_app_data_folder() / "taxonomy-v3" / _TAXONOMY_V3_DL_SHA256[:12]


def _taxonomy_v3_path() -> Path:
  return _taxonomy_v3_dir() / _TAXONOMY_V3_DL_URL.rsplit("/", 1)[-1]


def _legacy_taxonomy_v3_path() -> Path:
  return get_birdnet_app_data_folder() / "taxonomy_v3_0.csv"


def _taxonomy_v3_lock_dir() -> Path:
  return get_birdnet_app_data_folder() / ".taxonomy_v3_0.lock"


def get_taxonomy_v3_input() -> LabelInput:
  """The taxonomy as an input a generated label directory can record."""
  return LabelInput(
    path=_taxonomy_v3_path(),
    url=_TAXONOMY_V3_DL_URL,
    size=_TAXONOMY_V3_DL_SIZE,
    sha256=_TAXONOMY_V3_DL_SHA256,
  )


def taxonomy_v3_available() -> bool:
  """By content: a file of the right length from another release is not this one."""
  return artifact_is_current(get_taxonomy_v3_input())


def ensure_taxonomy_v3_available() -> Path:
  with directory_lock(_taxonomy_v3_lock_dir(), "the shared v3.0 taxonomy setup"):
    ensure_artifact(
      get_taxonomy_v3_input(),
      "Downloading shared v3.0 taxonomy",
      legacy_path=_legacy_taxonomy_v3_path(),
    )
  return _taxonomy_v3_path()
