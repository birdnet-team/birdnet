"""Process-wide progress callback for model, label and taxonomy downloads.

``birdnet.set_download_progress_callback(cb)`` (or the scoped
``download_progress_callback(cb)``) replaces the default stderr tqdm bar with
:class:`DownloadProgress` snapshots, per downloaded file:

- ``"started"`` once per attempt, before any I/O -- a repeat with a higher
  ``attempt`` and ``bytes_done == 0`` announces a retry;
- ``"progress"`` at most every 0.1 s (first chunk of an attempt always);
- ``"retrying"`` before each back-off, with ``error`` and ``retry_in_s``;
- exactly one of ``"finished"`` / ``"failed"`` (the error is raised right after).

A callback that raises aborts the download (partial file discarded, no retry,
no further events) and its exception propagates out of ``load(..)`` -- the way
to cancel from a UI. Calls are synchronous on the ``load(..)`` thread; the
callback is captured when a download starts. One ``load(..)`` may run several
downloads (labels, taxonomy, model): key on ``description``/``url``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

DownloadStatus = Literal["started", "progress", "retrying", "finished", "failed"]


@dataclass(frozen=True)
class DownloadProgress:
  """One update from a model/label/taxonomy download (see the module docstring)."""

  description: str
  url: str
  bytes_done: int
  bytes_total: int | None
  attempt: int
  max_attempts: int
  status: DownloadStatus
  error: str | None = None
  retry_in_s: float | None = None

  @property
  def fraction(self) -> float | None:
    """Progress in [0, 1], or ``None`` while the total size is unknown."""
    if self.bytes_total is None or self.bytes_total <= 0:
      return None
    return min(self.bytes_done / self.bytes_total, 1.0)

  @property
  def is_terminal(self) -> bool:
    return self.status in ("finished", "failed")


DownloadProgressCallback = Callable[[DownloadProgress], None]

_lock = threading.Lock()
_callback: DownloadProgressCallback | None = None

# Downloads stream in 1 KiB chunks, so a 520 MB model is ~530 000 chunks; the
# callback must not be invoked per chunk. 0.1 s is fast enough for any UI.
_PROGRESS_MIN_INTERVAL_S = 0.1
# Indirection so tests can drive the throttle with a fake clock without
# patching `time.monotonic` for the whole process (requests/urllib3 use it).
_monotonic = time.monotonic


def set_download_progress_callback(
  callback: DownloadProgressCallback | None,
) -> DownloadProgressCallback | None:
  """Register a process-wide callback for download progress; returns the previous one.

  Pass ``None`` to unregister. With no callback registered (the default),
  downloads behave exactly as before: a tqdm bar on stderr. While a callback is
  registered the tqdm bar is disabled. An exception raised by the callback
  aborts the running download without a retry and propagates out of
  ``load(..)`` (see the module docstring).
  """
  global _callback
  with _lock:
    previous = _callback
    _callback = callback
  return previous


def get_download_progress_callback() -> DownloadProgressCallback | None:
  with _lock:
    return _callback


@contextmanager
def download_progress_callback(
  callback: DownloadProgressCallback,
) -> Generator[None, None, None]:
  """Scoped alternative to :func:`set_download_progress_callback`.

  Registers ``callback`` for the duration of the ``with`` block and restores
  whatever was registered before on exit (including ``None``).
  """
  previous = set_download_progress_callback(callback)
  try:
    yield
  finally:
    set_download_progress_callback(previous)


class _ProgressThrottle:
  def __init__(self, min_interval_s: float) -> None:
    self._min_interval_s = min_interval_s
    self._last_reported_at: float | None = None

  def should_report(self) -> bool:
    now = _monotonic()
    if (
      self._last_reported_at is None
      or now - self._last_reported_at >= self._min_interval_s
    ):
      self._last_reported_at = now
      return True
    return False


class DownloadReporter:
  """Emits the events of one download (all attempts) to one captured callback.

  Internal helper for ``download_file_tqdm``; not part of the public API. With
  ``callback=None`` every method is a no-op, so the default path costs one
  attribute check per chunk.
  """

  def __init__(
    self,
    url: str,
    description: str | None,
    max_attempts: int,
    callback: DownloadProgressCallback | None,
  ) -> None:
    self.url = url
    self.description = description or url
    self.max_attempts = max_attempts
    self._callback = callback
    self.attempt = 0
    self.bytes_done = 0
    self.bytes_total: int | None = None
    self.callback_failed = False
    self._throttle = _ProgressThrottle(_PROGRESS_MIN_INTERVAL_S)

  @property
  def enabled(self) -> bool:
    return self._callback is not None

  def started(self, attempt: int, bytes_total: int | None) -> None:
    self.attempt = attempt
    self.bytes_done = 0
    self.bytes_total = bytes_total
    self._throttle = _ProgressThrottle(_PROGRESS_MIN_INTERVAL_S)
    self._emit("started")

  def total_known(self, bytes_total: int | None) -> None:
    # Called once the response headers are in; matters when no chunk follows
    # (empty file) so that "finished" still carries the total.
    self.bytes_total = bytes_total

  def progress(self, bytes_done: int) -> None:
    self.bytes_done = bytes_done
    if self._callback is not None and self._throttle.should_report():
      self._emit("progress")

  def finished(self) -> None:
    self._emit("finished")

  def retrying(self, error: BaseException, wait_s: float) -> None:
    self._emit("retrying", error=str(error), retry_in_s=wait_s)

  def failed(self, error: BaseException) -> None:
    self._emit("failed", error=str(error))

  def _emit(
    self,
    status: DownloadStatus,
    *,
    error: str | None = None,
    retry_in_s: float | None = None,
  ) -> None:
    if self._callback is None:
      return
    progress = DownloadProgress(
      description=self.description,
      url=self.url,
      bytes_done=self.bytes_done,
      bytes_total=self.bytes_total,
      attempt=self.attempt,
      max_attempts=self.max_attempts,
      status=status,
      error=error,
      retry_in_s=retry_in_s,
    )
    try:
      self._callback(progress)
    except BaseException:
      # Lets the retry loop tell a callback failure from a download failure:
      # the callback may raise ValueError, which the loop would otherwise retry.
      self.callback_failed = True
      raise
