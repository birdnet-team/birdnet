"""Shared steps of the official model and label downloads.

Each zip downloader says which archive members go where; fetching, extracting,
placing and the cross-process serialization live here, once. So does the
checksummed single-file download that models and labels share.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from birdnet.utils import helper
from birdnet.utils.helper import (
  check_protobuf_model_files_exist,
  check_source_marker,
  directory_lock,
  lock_heartbeat,
)

logger = logging.getLogger(__name__)


def saved_model_is_current(model_dir: Path, dl_url: str) -> bool:
  """Whether `model_dir` holds a complete SavedModel downloaded from `dl_url`."""
  return (
    model_dir.is_dir()
    and check_protobuf_model_files_exist(model_dir)
    and check_source_marker(model_dir, dl_url)
  )


def download_zip_members(
  url: str,
  download_size: int,
  description: str,
  members: Sequence[tuple[str | None, Path]],
) -> None:
  """Download the zip at `url` and move each member onto its destination.

  `members` pairs an archive path (`None` for the extraction root) with the
  path that is to hold it; whatever is at that path is replaced. Source markers
  are the caller's business. `description` names the model, for the progress
  bar and the log.
  """
  # Scratch space beside the destinations: each move is then a rename, so a kill
  # mid-way leaves no half-copied member behind.
  scratch_root = members[0][1].parent
  scratch_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
    prefix="birdnet_download", dir=scratch_root
  ) as temp_dir:
    zip_path = Path(temp_dir) / "download.zip"
    # Via the module: tests fake the transport there.
    helper.download_file_tqdm(
      url,
      zip_path,
      download_size=download_size,
      description=f"Downloading {description}",
    )

    logger.info(f"Extracting {description}...")
    extract_dir = Path(temp_dir) / "extracted"
    with zipfile.ZipFile(zip_path, "r") as archive:
      archive.extractall(extract_dir)

    for member, destination in members:
      source = extract_dir if member is None else extract_dir / member
      destination.parent.mkdir(parents=True, exist_ok=True)
      if destination.is_dir():
        shutil.rmtree(destination, ignore_errors=True)
      else:
        destination.unlink(missing_ok=True)
      shutil.move(source, destination)


def ensure_verified_file(
  path: Path,
  url: str,
  size: int,
  sha256: str,
  description: str,
  legacy_path: Path | None = None,
  *,
  hash_on_hit: bool,
) -> None:
  """Put the file `sha256` describes at `path`, downloading it if needed.

  Only verified bytes ever land at `path`. A file already there is a hit at
  `size` bytes, and with `hash_on_hit` only if its digest matches too. A
  matching file at `legacy_path` is adopted rather than fetched; one that hashes
  differently is left alone, for another installed version still reading it.
  """

  def is_current() -> bool:
    if not (path.is_file() and path.stat().st_size == size):
      return False
    return not hash_on_hit or helper.sha256_file(path) == sha256

  if is_current():
    return

  path.parent.mkdir(parents=True, exist_ok=True)
  if legacy_path is not None:
    try:
      if (
        legacy_path.is_file()
        and legacy_path.stat().st_size == size
        and helper.sha256_file(legacy_path) == sha256
      ):
        # Windows refuses this while another process has the file open; the
        # re-check below then sends this process to the download instead.
        os.replace(legacy_path, path)
        return
    except OSError:
      # The legacy file vanished mid-probe: a concurrent process adopted it.
      pass
    # Downloads are not serialized, so a concurrent process may have adopted or
    # downloaded the file while this one was probing; its bytes are the wanted
    # ones (only verified content ever lands at this path).
    if is_current():
      return

  # Downloaded next to the target and renamed only after verification, so a
  # process killed during the download or the hash cannot leave bytes nobody
  # checked at `path`. The scratch name is per-process; a shared one would let
  # concurrent downloads fail each other's rename on Windows.
  fd, temp_name = tempfile.mkstemp(
    dir=path.parent, prefix=f"{path.name}.", suffix=".unverified"
  )
  os.close(fd)
  temp_path = Path(temp_name)
  try:
    # Via the module: tests fake the transport and the hash there.
    helper.download_file_tqdm(
      url, temp_path, download_size=size, description=description
    )
    actual = helper.sha256_file(temp_path)
    if actual != sha256:
      raise RuntimeError(
        f"The file downloaded from {url} does not match its expected "
        f"checksum ({actual} instead of {sha256}). It was discarded; "
        "retry, and if this persists the published file has changed."
      )
    try:
      temp_path.replace(path)
    except OSError:
      # Windows refuses this while a concurrent process still holds the file it
      # just published open. A file there that passes the hit check holds the
      # wanted bytes, so this one's are redundant.
      if not is_current():
        raise
  finally:
    temp_path.unlink(missing_ok=True)


def ensure_download(
  available: Callable[[], bool],
  download: Callable[[], None],
  lock_dir: Path,
  description: str,
  *,
  timeout_s: float = 300.0,
) -> None:
  """Run `download()` unless `available()`, once across concurrent processes.

  A process that finds the model missing takes the lock and checks again before
  downloading, so a download another process finished meanwhile is adopted
  rather than repeated, and one still in progress is never torn down under it.
  The download beats the lock while it runs, so `timeout_s` bounds the wait on
  a holder whose process is gone, not the download itself; a holder that is
  alive is waited on, and a stalled transfer is ended by the transport's own
  request timeout and bounded retries.
  """
  # Checked before the lock: a current cache then needs no write at all, which
  # keeps read-only and pre-populated app data directories usable.
  if available():
    return
  with directory_lock(lock_dir, description, timeout_s=timeout_s):
    if available():
      return
    with lock_heartbeat(lock_dir, interval_s=timeout_s / 10):
      download()
