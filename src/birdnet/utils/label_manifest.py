"""Records what a generated label directory was actually built from.

The V3.0 label files are generated locally from two downloads: the model's own
label file and the taxonomy shared by the acoustic and geo models. Deciding
whether they are current from the *constants the running version holds* is not
enough - it says which release was meant to be read, never which one was. Two
installed versions sharing one app data directory is enough to make those differ.

So a manifest records observations: the digest of the bytes actually read, and
the digest of every file actually written. Verification compares both against
the running version's constants and against what is on disk now, so a swapped
input or an edited output is caught whatever produced it.

Hashing on every load would be wasteful, so each entry also carries `(size,
mtime_ns)`. Matching stat means matching content in practice, and only a stat
mismatch falls through to the digest - which keeps a restored CI cache or a copy
that lost its timestamps cheap instead of forcing a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from birdnet.utils.helper import download_file_tqdm, write_text_atomic
from birdnet.utils.logging_utils import get_logger_for_package

MANIFEST_NAME = ".birdnet_labels.json"

# Bumped when the manifest layout itself changes; an unknown value is treated as
# "not up to date" rather than parsed on a guess.
MANIFEST_VERSION = 1

# Written by releases before the manifest existed. Removed when a directory is
# regenerated so it cannot be mistaken for a current record later.
LEGACY_MARKER_NAME = ".birdnet_taxonomy"

_LANG_SUFFIX = ".txt"


@dataclass(frozen=True)
class LabelInput:
  """A downloaded artifact a generated label directory was built from."""

  path: Path
  url: str
  size: int
  sha256: str


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def get_manifest_path(lang_dir: Path) -> Path:
  return lang_dir / MANIFEST_NAME


def verify_download(path: Path, expected: LabelInput) -> None:
  """Check a freshly downloaded artifact against its pinned digest.

  Byte size alone cannot tell two releases apart when they happen to be the same
  length, and it is the size check that let a different taxonomy take this path.
  A file that fails is removed, so a retry starts from nothing.
  """
  actual = sha256_file(path)
  if actual == expected.sha256:
    return
  path.unlink(missing_ok=True)
  raise RuntimeError(
    f"The file downloaded from {expected.url} does not match its expected "
    f"checksum ({actual} instead of {expected.sha256}). It was discarded; "
    "retry, and if this persists the published file has changed."
  )


def artifact_is_current(expected: LabelInput) -> bool:
  """Whether the file on disk is the artifact these constants describe.

  By content, not by byte size: two releases can share a length, and deciding
  this by size is what let one version's taxonomy stand in for another's.
  """
  if not expected.path.is_file():
    return False
  if expected.path.stat().st_size != expected.size:
    return False
  return sha256_file(expected.path) == expected.sha256


def ensure_artifact(
  expected: LabelInput, description: str, legacy_path: Path | None = None
) -> None:
  """Put the artifact these constants describe on disk, downloading if needed.

  A file left at `legacy_path` by an earlier layout is adopted rather than
  fetched again, so upgrading stays offline for anyone already holding it. One
  that hashes differently is left alone: it belongs to another installed version
  still reading it from there.
  """
  if artifact_is_current(expected):
    return

  expected.path.parent.mkdir(parents=True, exist_ok=True)
  if (
    legacy_path is not None
    and legacy_path.is_file()
    and legacy_path.stat().st_size == expected.size
    and sha256_file(legacy_path) == expected.sha256
  ):
    with suppress(OSError):
      # Windows refuses this while another process has the file open; falling
      # through to the download is correct, just slower.
      os.replace(legacy_path, expected.path)
      return

  download_file_tqdm(
    expected.url,
    expected.path,
    download_size=expected.size,
    description=description,
  )
  verify_download(expected.path, expected)


def _stat_of(path: Path) -> dict[str, int]:
  st = path.stat()
  return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _matches_on_disk(path: Path, entry: dict[str, Any], expected_sha256: str) -> bool:
  """Whether `path` still holds the recorded content.

  Returns True on a stat match without hashing; a stat mismatch is not yet a
  failure, because a cache restore or a plain copy rewrites mtime.
  """
  if not path.is_file():
    return False
  try:
    st = _stat_of(path)
  except OSError:
    return False
  if st["size"] != entry.get("size"):
    return False
  if st["mtime_ns"] == entry.get("mtime_ns"):
    return True
  return sha256_file(path) == expected_sha256


def write_manifest(
  lang_dir: Path,
  *,
  generator: str,
  generator_version: int,
  inputs: dict[str, LabelInput],
  languages: dict[str, str],
  lang_files: list[Path],
  stats: dict[str, int] | None = None,
) -> None:
  """Record what this directory was built from. Written last, atomically."""
  manifest: dict[str, Any] = {
    "manifest_version": MANIFEST_VERSION,
    "generator": generator,
    "generator_version": generator_version,
    "inputs": {
      name: {
        "url": inp.url,
        "size": inp.size,
        "sha256": inp.sha256,
        **_stat_of(inp.path),
      }
      for name, inp in inputs.items()
    },
    "languages": dict(languages),
    "files": {
      f.name: {"sha256": sha256_file(f), **_stat_of(f)} for f in sorted(lang_files)
    },
    "stats": dict(stats or {}),
  }
  _write_json_atomic(get_manifest_path(lang_dir), manifest)


def read_manifest(lang_dir: Path) -> dict[str, Any] | None:
  path = get_manifest_path(lang_dir)
  if not path.is_file():
    return None
  try:
    loaded = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return None
  return loaded if isinstance(loaded, dict) else None


def labels_up_to_date(
  lang_dir: Path,
  *,
  generator: str,
  generator_version: int,
  inputs: dict[str, LabelInput],
  languages: dict[str, str],
) -> bool:
  """Whether `lang_dir` holds files this version would generate right now."""
  manifest = read_manifest(lang_dir)
  if manifest is None:
    return False
  if manifest.get("manifest_version") != MANIFEST_VERSION:
    return False
  if manifest.get("generator") != generator:
    return False
  if manifest.get("generator_version") != generator_version:
    return False
  # Exact comparison, so adding, removing or remapping a language all count.
  if manifest.get("languages") != languages:
    return False

  recorded_inputs = manifest.get("inputs")
  if not isinstance(recorded_inputs, dict) or set(recorded_inputs) != set(inputs):
    return False
  for name, inp in inputs.items():
    entry = recorded_inputs.get(name)
    if not isinstance(entry, dict):
      return False
    if (
      entry.get("url") != inp.url
      or entry.get("size") != inp.size
      or entry.get("sha256") != inp.sha256
    ):
      return False
    if not _matches_on_disk(inp.path, entry, inp.sha256):
      return False

  recorded_files = manifest.get("files")
  if not isinstance(recorded_files, dict):
    return False
  # Extras count as stale too: a language dropped upstream must not survive here.
  on_disk = {p.name for p in lang_dir.glob(f"*{_LANG_SUFFIX}")}
  if on_disk != set(recorded_files):
    return False
  if {f"{lang}{_LANG_SUFFIX}" for lang in languages} != on_disk:
    return False
  for name, entry in recorded_files.items():
    if not isinstance(entry, dict):
      return False
    recorded_sha = entry.get("sha256")
    if not isinstance(recorded_sha, str):
      return False
    if not _matches_on_disk(lang_dir / name, entry, recorded_sha):
      return False

  _refresh_stats(lang_dir, manifest, inputs)
  return True


def prune_stale_entries(lang_dir: Path, keep: set[str]) -> None:
  """Drop label files this version no longer produces, and the legacy marker."""
  for path in lang_dir.glob(f"*{_LANG_SUFFIX}"):
    if path.name not in keep:
      path.unlink(missing_ok=True)
  (lang_dir / LEGACY_MARKER_NAME).unlink(missing_ok=True)


def record_generation(
  lang_dir: Path,
  *,
  generator: str,
  generator_version: int,
  declared_inputs: dict[str, LabelInput],
  read_bytes: dict[str, bytes],
  languages: dict[str, str],
  lang_files: list[Path],
  stats: dict[str, int] | None = None,
) -> None:
  """Close out a generation: drop what is no longer produced, then record it.

  `read_bytes` is what the generator actually parsed, keyed like
  `declared_inputs`. Recording the digest of *those* bytes rather than the
  pinned constant is the whole point - it is what later reveals that the file on
  a shared path was not the release this directory was told to expect.
  """
  prune_stale_entries(lang_dir, keep={f.name for f in lang_files})
  write_manifest(
    lang_dir,
    generator=generator,
    generator_version=generator_version,
    inputs={
      name: replace(declared, sha256=sha256_bytes(read_bytes[name]))
      for name, declared in declared_inputs.items()
    },
    languages=languages,
    lang_files=lang_files,
    stats=stats,
  )


def _refresh_stats(
  lang_dir: Path, manifest: dict[str, Any], inputs: dict[str, LabelInput]
) -> None:
  """Re-record `(size, mtime_ns)` after a digest confirmed the content.

  Without this, a directory whose timestamps changed once - a restored cache, a
  copy without `-p` - would hash every input and every output on every load.
  Best effort: a read-only cache stays usable, just slower.
  """
  changed = False
  for name, inp in inputs.items():
    entry = manifest["inputs"][name]
    try:
      current = _stat_of(inp.path)
    except OSError:
      return
    if current != {"size": entry.get("size"), "mtime_ns": entry.get("mtime_ns")}:
      entry.update(current)
      changed = True
  for name, entry in manifest["files"].items():
    try:
      current = _stat_of(lang_dir / name)
    except OSError:
      return
    if current != {"size": entry.get("size"), "mtime_ns": entry.get("mtime_ns")}:
      entry.update(current)
      changed = True

  if not changed:
    return
  try:
    _write_json_atomic(get_manifest_path(lang_dir), manifest)
  except OSError as err:
    get_logger_for_package(__name__).debug(
      f"Could not refresh the label manifest in {lang_dir}: {err}"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
  write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))
