"""ensure_download(..): one download per model, whoever gets to it first."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from birdnet.utils import downloads
from birdnet.utils.downloads import ensure_download
from birdnet.utils.helper import directory_lock

pytestmark = pytest.mark.no_tf


class _Cache:
  """A model cache whose availability flips once its download has run."""

  def __init__(self, available: bool = False) -> None:
    self.available = available
    self.downloads = 0
    self.checked = threading.Event()

  def check(self) -> bool:
    self.checked.set()
    return self.available

  def download(self) -> None:
    self.downloads += 1
    self.available = True


def test_an_available_model_takes_neither_lock_nor_download(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  cache = _Cache(available=True)
  locks_taken: list[Path] = []
  real_lock = downloads.directory_lock

  def recording_lock(lock_dir: Path, *args: object, **kwargs: object) -> object:
    locks_taken.append(lock_dir)
    return real_lock(lock_dir, *args, **kwargs)  # type: ignore[arg-type]

  monkeypatch.setattr(downloads, "directory_lock", recording_lock)

  ensure_download(cache.check, cache.download, tmp_path / ".lock", "a model")

  assert cache.downloads == 0
  assert locks_taken == []


def test_a_missing_model_is_downloaded_once(tmp_path: Path) -> None:
  cache = _Cache()
  lock_dir = tmp_path / ".lock"

  ensure_download(cache.check, cache.download, lock_dir, "a model")

  assert cache.downloads == 1
  assert not lock_dir.exists()


def test_a_model_the_lock_holder_provides_is_not_downloaded_again(
  tmp_path: Path,
) -> None:
  """The waiter found the model missing, then waited for the lock; the holder
  downloaded it meanwhile. The waiter must re-check and adopt that result."""
  cache = _Cache()
  lock_dir = tmp_path / ".lock"
  waiter = threading.Thread(
    target=ensure_download,
    args=(cache.check, cache.download, lock_dir, "a model"),
    daemon=True,
  )

  with directory_lock(lock_dir, "the holder's download"):
    waiter.start()
    assert cache.checked.wait(timeout=10)
    cache.available = True

  waiter.join(timeout=30)
  assert not waiter.is_alive()
  assert cache.downloads == 0


def test_a_failing_download_releases_the_lock(tmp_path: Path) -> None:
  lock_dir = tmp_path / ".lock"

  def failing_download() -> None:
    raise RuntimeError("boom")

  with pytest.raises(RuntimeError, match="boom"):
    ensure_download(lambda: False, failing_download, lock_dir, "a model")

  assert not lock_dir.exists()

  cache = _Cache()
  ensure_download(cache.check, cache.download, lock_dir, "a model")
  assert cache.downloads == 1


def test_a_waiter_outlasts_the_timeout_while_the_holder_downloads(
  tmp_path: Path,
) -> None:
  """The timeout bounds silence from the holder, not the download: a second
  process waits out a download longer than the timeout and adopts its result."""
  cache = _Cache()
  lock_dir = tmp_path / ".lock"

  def slow_download() -> None:
    time.sleep(1.5)
    cache.download()

  holder = threading.Thread(
    target=ensure_download,
    args=(cache.check, slow_download, lock_dir, "a model"),
    kwargs={"timeout_s": 0.5},
    daemon=True,
  )
  holder.start()
  deadline = time.monotonic() + 10
  while not lock_dir.is_dir():
    assert time.monotonic() < deadline, "the holder never took the lock"
    time.sleep(0.01)

  ensure_download(cache.check, cache.download, lock_dir, "a model", timeout_s=0.5)

  holder.join(10)
  assert not holder.is_alive()
  assert cache.downloads == 1
