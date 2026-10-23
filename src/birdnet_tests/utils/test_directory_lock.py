import os
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

from birdnet.utils.helper import directory_lock, lock_heartbeat


@pytest.mark.no_tf
def test_releases_the_lock_afterwards(tmp_path: Path) -> None:
  lock_dir = tmp_path / ".lock"

  with directory_lock(lock_dir, "a test"):
    assert lock_dir.is_dir()

  assert not lock_dir.exists()


@pytest.mark.no_tf
def test_releases_the_lock_when_the_block_raises(tmp_path: Path) -> None:
  lock_dir = tmp_path / ".lock"

  with pytest.raises(RuntimeError, match="boom"), directory_lock(lock_dir, "a test"):
    raise RuntimeError("boom")

  assert not lock_dir.exists()


@pytest.mark.no_tf
def test_releases_the_lock_while_a_waiter_reads_the_owner_file(
  tmp_path: Path,
) -> None:
  """A waiter has `owner` open for the moment it reads the pid, and Windows
  refuses to delete an open file. A release that gives up on that moment leaves
  the lock behind with the holder's live pid inside, and nobody reclaims it.
  Elsewhere the delete goes through regardless, so this only bites on Windows.
  """
  lock_dir = tmp_path / ".lock"

  with directory_lock(lock_dir, "a test"):
    reader = (lock_dir / "owner").open(encoding="utf-8")
    threading.Timer(0.2, reader.close).start()

  assert not lock_dir.exists()


@pytest.mark.no_tf
def test_release_leaves_a_lock_a_waiter_took_in_the_meantime(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Once the directory is gone the release is over: a waiter may already hold
  the lock, and removing the directory again would take it from them."""
  lock_dir = tmp_path / ".lock"
  real_rmtree = shutil.rmtree
  waiter_took_it = False

  def rmtree_then_a_waiter_takes_the_lock(
    path: Path, *args: object, **kw: object
  ) -> None:
    nonlocal waiter_took_it
    real_rmtree(path, *args, **kw)
    if not waiter_took_it and not Path(path).exists():
      Path(path).mkdir()
      (Path(path) / "owner").write_text("424242", encoding="utf-8")
      waiter_took_it = True

  monkeypatch.setattr(shutil, "rmtree", rmtree_then_a_waiter_takes_the_lock)
  with directory_lock(lock_dir, "a test"):
    pass

  assert waiter_took_it
  assert (lock_dir / "owner").read_text(encoding="utf-8") == "424242", (
    "the waiter's lock was removed"
  )


@pytest.mark.no_tf
def test_times_out_while_a_live_process_holds_it(tmp_path: Path) -> None:
  lock_dir = tmp_path / ".lock"
  lock_dir.mkdir()
  # This process is alive, so the lock is not abandoned.
  (lock_dir / "owner").write_text(str(os.getpid()), encoding="utf-8")

  started = time.monotonic()
  with (
    pytest.raises(TimeoutError, match="a test"),
    directory_lock(lock_dir, "a test", timeout_s=0.3),
  ):
    pytest.fail("the lock must not be granted while it is held")

  assert time.monotonic() - started >= 0.3
  assert lock_dir.is_dir(), "a live holder's lock must be left alone"


@pytest.mark.no_tf
def test_the_error_names_the_directory_to_remove(tmp_path: Path) -> None:
  """A user who hits this has to be able to find what is holding it."""
  lock_dir = tmp_path / ".lock"
  lock_dir.mkdir()
  (lock_dir / "owner").write_text(str(os.getpid()), encoding="utf-8")

  with (
    pytest.raises(TimeoutError, match=re.escape(str(lock_dir))),
    directory_lock(lock_dir, "a test", timeout_s=0.1),
  ):
    pass


@pytest.mark.no_tf
def test_reclaims_a_lock_left_by_a_process_that_no_longer_exists(
  tmp_path: Path,
) -> None:
  """A crash mid-setup must not make the library permanently unusable.

  The same kill that leaks the lock is the one that invalidates the cache, so
  without reclaiming, every later load waits out the timeout and fails for good.
  """
  lock_dir = tmp_path / ".lock"
  lock_dir.mkdir()
  dead_pid = _a_pid_that_does_not_exist()
  (lock_dir / "owner").write_text(str(dead_pid), encoding="utf-8")

  started = time.monotonic()
  with directory_lock(lock_dir, "a test", timeout_s=5):
    assert lock_dir.is_dir()

  assert time.monotonic() - started < 1, "it must not wait out the timeout"
  assert not lock_dir.exists()


def _a_pid_that_does_not_exist() -> int:
  import psutil

  for candidate in range(60000, 65000):
    if not psutil.pid_exists(candidate):
      return candidate
  raise AssertionError("no free pid found")


def _wait_until_held(lock_dir: Path) -> None:
  deadline = time.monotonic() + 10
  while not lock_dir.is_dir():
    assert time.monotonic() < deadline, "the holder never took the lock"
    time.sleep(0.01)


@pytest.mark.no_tf
def test_a_waiter_outlasts_its_timeout_while_the_holder_beats(tmp_path: Path) -> None:
  """The timeout bounds the time since the holder's last beat, not the whole
  wait: a holder in a step longer than the timeout keeps its waiters."""
  lock_dir = tmp_path / ".lock"
  release = threading.Event()

  def hold() -> None:
    with directory_lock(lock_dir, "a test"), lock_heartbeat(lock_dir, interval_s=0.05):
      release.wait(10)

  holder = threading.Thread(target=hold, daemon=True)
  holder.start()
  _wait_until_held(lock_dir)
  threading.Timer(1.5, release.set).start()

  with directory_lock(lock_dir, "a test", timeout_s=0.5):
    pass

  holder.join(10)
  assert not holder.is_alive()


@pytest.mark.no_tf
def test_a_waiter_gives_up_once_the_beat_is_stale(tmp_path: Path) -> None:
  """A beat that stops changing is given up on after `timeout_s` of the
  waiter's own time, whatever the file's timestamp says."""
  lock_dir = tmp_path / ".lock"
  with directory_lock(lock_dir, "a test"):
    (lock_dir / "heartbeat").touch()

    started = time.monotonic()
    with (
      pytest.raises(TimeoutError, match="waiting for a test"),
      directory_lock(lock_dir, "a test", timeout_s=0.5),
    ):
      pass
    elapsed = time.monotonic() - started

  assert 0.5 <= elapsed < 10


@pytest.mark.no_tf
def test_a_beat_stamped_by_a_clock_behind_ours_is_not_stale(tmp_path: Path) -> None:
  """On a network share the beat's mtime is the file server's clock. A server
  an hour behind must not make every fresh beat look an hour old."""
  lock_dir = tmp_path / ".lock"
  release = threading.Event()

  def hold_and_beat_an_hour_behind() -> None:
    with directory_lock(lock_dir, "a test"):
      beat = lock_dir / "heartbeat"
      while not release.wait(0.05):
        beat.touch()
        behind = time.time() - 3600
        os.utime(beat, (behind, behind))

  holder = threading.Thread(target=hold_and_beat_an_hour_behind, daemon=True)
  holder.start()
  _wait_until_held(lock_dir)
  threading.Timer(1.5, release.set).start()

  with directory_lock(lock_dir, "a test", timeout_s=0.5):
    pass

  holder.join(10)
  assert not holder.is_alive()
