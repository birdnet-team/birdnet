"""Official model and label downloads: the transport and the shared steps.

`download_file_tqdm` fetches a URL with retries and progress reporting. Each zip
downloader says which archive members go where; fetching, extracting, placing
and the cross-process serialization live here, once. So do the checksummed
single-file download that models and labels share, the `ModelInfo` records that
describe a download, and the source markers that tell SavedModel releases apart.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from birdnet.utils.download_progress import (
  DownloadReporter,
  get_download_progress_callback,
)
from birdnet.utils.filesystem import directory_lock, lock_heartbeat

logger = logging.getLogger(__name__)

# Each block costs fixed Python work, about half of it inside requests' chunk
# delivery and the rest in this loop (tqdm update, write, progress call): 1 KiB
# blocks capped a download from a local server below 200 MB/s, 64 KiB blocks
# run several times faster, and larger blocks save only a fraction of a second
# per model.
_DOWNLOAD_BLOCK_SIZE = 64 * 1024


def check_protobuf_model_files_exist(folder: Path) -> bool:
  exists = True
  exists &= (folder / "saved_model.pb").is_file()
  exists &= (folder / "variables").is_dir()
  exists &= (folder / "variables" / "variables.data-00000-of-00001").is_file()
  exists &= (folder / "variables" / "variables.index").is_file()
  return exists


# Records which release a SavedModel directory came from; its name is generic,
# so nothing else tells releases apart.
SOURCE_MARKER_NAME = ".birdnet_source"


def write_source_marker(model_dir: Path, dl_url: str) -> None:
  (model_dir / SOURCE_MARKER_NAME).write_text(dl_url, encoding="utf-8")


def check_source_marker(model_dir: Path, dl_url: str) -> bool:
  marker = model_dir / SOURCE_MARKER_NAME
  if not marker.is_file():
    return False
  return marker.read_text(encoding="utf-8").strip() == dl_url


@dataclass()
class ModelInfo:
  dl_url: str
  dl_size: int
  file_size: int
  dl_file_name: str
  # Unset only for the zip-packaged v2.4 downloads, which are still judged by
  # extracted byte size alone.
  sha256: str | None = None

  @property
  def content_tag(self) -> str:
    """Checksum prefix embedded in the cached file's name.

    The name then identifies the release content itself: a release that expects
    different bytes looks for a different file, so it can neither serve another
    release's model nor overwrite it in a shared app data directory."""
    assert self.sha256 is not None
    return self.sha256[:12]


# Transient network faults (connection resets, read timeouts, truncated
# streams, 5xx) must not fail a model/label download outright: every official
# model goes through `download_file_tqdm`, so a single fault would otherwise
# surface as a failed `load()`. Client errors (4xx) are permanent and are raised
# at once.
#
# The back-off spans ~110 s in total because the observed failure mode is not a
# single dropped packet: GitHub's release-download endpoint refuses connections
# ("Remote end closed connection without response") for tens of seconds at a
# time. A 20 s window was measured in CI to be too short -- a sibling step
# retrying the same host after 30 s succeeded on its second try while
# `download_file_tqdm` exhausted three attempts. The wait is only ever paid on a
# download that is already failing.
_DOWNLOAD_ATTEMPTS = 5
_DOWNLOAD_RETRY_WAITS_S = (5.0, 15.0, 30.0, 60.0)


class DownloadError(ValueError):
  """A download did not complete successfully.

  A ``ValueError``, so callers that catch one for a failed download keep working;
  ``status_code`` lets callers and the retry loop in ``download_file_tqdm`` tell
  a permanent client error from a retriable one.
  """

  def __init__(self, message: str, *, status_code: int | None = None) -> None:
    super().__init__(message)
    self.status_code = status_code


def _is_retriable_download_error(error: Exception) -> bool:
  import requests

  status: int | None = None
  if isinstance(error, DownloadError):
    status = error.status_code
  elif isinstance(error, requests.HTTPError) and error.response is not None:
    status = error.response.status_code

  # 4xx means we asked for something that does not exist or may not be read;
  # repeating the request cannot change that. Everything else -- connection
  # resets, timeouts, truncated streams, 5xx -- is worth another attempt.
  is_client_error = status is not None and 400 <= status < 500
  return not is_client_error


def download_file_tqdm(
  url: str,
  file_path: Path,
  *,
  download_size: int | None = None,
  description: str | None = None,
) -> int:
  import requests

  # The callback is captured once per download, so a registration that changes
  # mid-way takes effect from the next download on (and the tqdm bar's on/off
  # state stays consistent with the events for this one).
  reporter = DownloadReporter(
    url, description, _DOWNLOAD_ATTEMPTS, get_download_progress_callback()
  )
  attempt = 0
  try:
    while True:
      attempt += 1
      try:
        return _download_file_once(
          url,
          file_path,
          download_size=download_size,
          description=description,
          attempt=attempt,
          reporter=reporter,
        )
      except (requests.RequestException, ValueError) as error:
        # Re-raise from inside the handler so the original traceback survives.
        if reporter.callback_failed:
          # The callback raised (a ValueError lands here too): abort, no retry.
          raise
        if attempt >= _DOWNLOAD_ATTEMPTS or not _is_retriable_download_error(error):
          raise
        wait_s = _DOWNLOAD_RETRY_WAITS_S[
          min(attempt - 1, len(_DOWNLOAD_RETRY_WAITS_S) - 1)
        ]
        # Before the log line: a callback that raises here cancels the download,
        # and then nothing is retried.
        reporter.retrying(error, wait_s)
        logging.getLogger(__name__).warning(
          f"Download of {url} failed (attempt {attempt}/{_DOWNLOAD_ATTEMPTS}): "
          f"{error}. Retrying in {wait_s:.0f} s..."
        )
        time.sleep(wait_s)
  except BaseException as error:
    # Terminal event for errors the retry handler does not deal in (OSError,
    # KeyboardInterrupt). A raising callback gets no further events; suppress
    # keeps one that raises again from masking the original error.
    if not reporter.callback_failed:
      with suppress(BaseException):
        reporter.failed(error)
    raise


def _download_file_once(
  url: str,
  file_path: Path,
  *,
  download_size: int | None = None,
  description: str | None = None,
  attempt: int = 1,
  reporter: DownloadReporter | None = None,
) -> int:
  assert file_path.parent.is_dir()
  import requests

  if reporter is None:
    reporter = DownloadReporter(url, description, _DOWNLOAD_ATTEMPTS, callback=None)

  reporter.started(attempt, download_size)

  response = requests.get(url, stream=True, timeout=120)
  try:
    content_length = response.headers.get("content-length")
    if download_size is not None:
      total_size = download_size
    elif content_length is not None:
      total_size = int(content_length)
    else:
      total_size = 0
    # 0 is the internal "unknown" sentinel (it also disables the size check).
    bytes_total: int | None = total_size
    if download_size is None and content_length is None:
      bytes_total = None
    reporter.total_known(bytes_total)

    fd, temp_name = tempfile.mkstemp(
      dir=file_path.parent,
      prefix=f"{file_path.name}.",
      suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)

    downloaded_size = 0
    try:
      with (
        tqdm(
          total=total_size,
          unit="iB",
          unit_scale=True,
          desc=description,
          disable=reporter.enabled,
        ) as tqdm_bar,
        open(temp_path, "wb") as file,
      ):
        for data in response.iter_content(_DOWNLOAD_BLOCK_SIZE):
          # Bytes are counted here rather than read from `tqdm_bar.n`, which
          # does not advance while the bar is disabled.
          tqdm_bar.update(len(data))
          file.write(data)
          downloaded_size += len(data)
          reporter.progress(downloaded_size)

      if response.status_code != 200 or (total_size not in (0, downloaded_size)):
        raise DownloadError(
          f"Failed to download the file. Status code: {response.status_code}\n"
          f"Expected size: {total_size} bytes, "
          f"downloaded size: {downloaded_size} bytes.",
          status_code=response.status_code,
        )

      temp_path.replace(file_path)
    except BaseException:
      # BaseException: a KeyboardInterrupt, or a callback raising to cancel,
      # must not leave the partial file behind either.
      temp_path.unlink(missing_ok=True)
      raise
  finally:
    response.close()

  reporter.finished()
  return total_size


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


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
    download_file_tqdm(
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
    return not hash_on_hit or sha256_file(path) == sha256

  if is_current():
    return

  path.parent.mkdir(parents=True, exist_ok=True)
  if legacy_path is not None:
    try:
      if (
        legacy_path.is_file()
        and legacy_path.stat().st_size == size
        and sha256_file(legacy_path) == sha256
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
    download_file_tqdm(url, temp_path, download_size=size, description=description)
    actual = sha256_file(temp_path)
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


def ensure_single_file_model(
  info: ModelInfo, model_path: Path, legacy_path: Path, description: str
) -> None:
  """Put the model file `info` describes at `model_path`, downloading if needed.

  A hit is by byte size alone: `model_path` carries `info.content_tag`, so a
  file there at the declared size is the one this release published. Hashing it
  on every load would cost seconds for the ~540 MB acoustic models. Byte size
  alone is not identity (see label_manifest.artifact_is_current).
  """
  assert info.sha256 is not None
  ensure_verified_file(
    model_path,
    info.dl_url,
    info.file_size,
    info.sha256,
    description,
    legacy_path,
    hash_on_hit=False,
  )


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
