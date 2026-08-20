import os
import re
import time
from pathlib import Path

import pytest

from birdnet.utils.helper import directory_lock


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
