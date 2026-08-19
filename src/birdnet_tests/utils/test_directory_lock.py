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
def test_times_out_while_another_holder_keeps_it(tmp_path: Path) -> None:
  """A process killed while holding it leaves the directory behind, so the wait
  is bounded rather than indefinite."""
  lock_dir = tmp_path / ".lock"
  lock_dir.mkdir()

  started = time.monotonic()
  with (
    pytest.raises(TimeoutError, match="a test"),
    directory_lock(lock_dir, "a test", timeout_s=0.3),
  ):
    pytest.fail("the lock must not be granted while it is held")

  assert time.monotonic() - started >= 0.3
  # the surviving directory belongs to the other holder and must be left alone
  assert lock_dir.is_dir()
