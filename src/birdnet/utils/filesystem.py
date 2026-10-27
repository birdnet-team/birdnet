"""Atomic file writes and the cross-process directory lock."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path


def write_text_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
  """Write via a temporary file in the same directory, then rename over.

  A reader either sees the previous content or the new one, never a partial
  file, and an interrupted write leaves nothing behind but a `.tmp` scratch file.

  Newlines are written through unchanged, so the same content yields the same
  bytes on every platform. Without that, files generated on Windows and on Linux
  differ, and anything that records a checksum of them records a different one
  per operating system.
  """
  fd, temp_name = tempfile.mkstemp(
    dir=path.parent,
    prefix=f"{path.name}.",
    suffix=".tmp",
  )
  os.close(fd)
  temp_path = Path(temp_name)

  try:
    temp_path.write_text(content, encoding=encoding, newline="\n")
    temp_path.replace(path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise


_LOCK_OWNER_NAME = "owner"

# A lock whose owner never wrote its pid is only reclaimed once it is old enough
# that the owner cannot still be between mkdir and the write.
_LOCK_ADOPTION_GRACE_S = 30.0


def _reclaim_if_abandoned(lock_dir: Path) -> bool:
  """Drop a lock whose owner is gone. Returns whether anything was reclaimed.

  Without this a process killed mid-setup - force-quit, OOM, power loss - leaves
  the directory behind and every later call waits out the timeout and fails, for
  good, with nothing to do but delete a hidden directory by hand.
  """
  import psutil

  owner = lock_dir / _LOCK_OWNER_NAME
  try:
    pid = int(owner.read_text(encoding="utf-8").strip())
  except (OSError, ValueError):
    try:
      age = time.time() - lock_dir.stat().st_mtime
    except OSError:
      return False
    if age < _LOCK_ADOPTION_GRACE_S:
      return False
  else:
    if psutil.pid_exists(pid):
      return False

  shutil.rmtree(lock_dir, ignore_errors=True)
  return not lock_dir.exists()


@contextmanager
def directory_lock(
  lock_dir: Path, description: str, timeout_s: float = 300.0
) -> Generator[None, None, None]:
  """Serialize one-time setup across processes by creating a directory.

  `mkdir` is atomic on every platform this runs on, which a lock file is not.
  The holder records its pid inside, so a lock left behind by a process that no
  longer exists is reclaimed rather than waited out.

  A holder in a long step keeps a beat going with `lock_heartbeat`;
  `timeout_s` then bounds the time since the beat last changed, not the wait
  as a whole.
  """
  deadline = time.monotonic() + timeout_s
  last_beat: int | None = None
  while True:
    try:
      lock_dir.mkdir(parents=True, exist_ok=False)
      break
    except FileExistsError as err:
      if _reclaim_if_abandoned(lock_dir):
        continue
      beat = _last_heartbeat(lock_dir)
      if beat is not None and beat != last_beat:
        # Timed on this process's clock from the moment the beat was seen to
        # change: the mtime itself is the file server's clock on a network
        # share and compares with nothing here.
        last_beat = beat
        deadline = time.monotonic() + timeout_s
      if time.monotonic() >= deadline:
        raise TimeoutError(
          f"Timed out while waiting for {description}. Another process is "
          f"holding {lock_dir}; if none is running, remove that directory."
        ) from err
      time.sleep(0.1)

  with suppress(OSError):
    (lock_dir / _LOCK_OWNER_NAME).write_text(str(os.getpid()), encoding="utf-8")
  try:
    yield
  finally:
    _release_lock(lock_dir)


# Windows refuses to delete a file another process has open, and a waiter has
# `owner` open for the moment it reads the pid. Retried for far longer than
# that moment lasts.
_LOCK_RELEASE_ATTEMPTS = 50
_LOCK_RELEASE_WAIT_S = 0.02


def _release_lock(lock_dir: Path) -> None:
  """Remove the lock directory; left behind, it looks held by a live process.

  Its owner pid stays valid as long as the holder runs, so no waiter would ever
  reclaim it. If the directory cannot be removed, the pid is blanked instead,
  which lets waiters adopt the lock after the grace period.
  """
  for _ in range(_LOCK_RELEASE_ATTEMPTS):
    shutil.rmtree(lock_dir, ignore_errors=True)
    # Gone, or a waiter's by now: removing it again would take theirs.
    if not _lock_is_held_by_me(lock_dir):
      return
    time.sleep(_LOCK_RELEASE_WAIT_S)
  logging.getLogger(__name__).warning(
    f"Could not remove the lock directory {lock_dir}; it will be adopted after "
    f"{_LOCK_ADOPTION_GRACE_S:.0f} s."
  )
  if _lock_is_held_by_me(lock_dir):
    with suppress(OSError):
      (lock_dir / _LOCK_OWNER_NAME).write_text("", encoding="utf-8")


def _lock_is_held_by_me(lock_dir: Path) -> bool:
  try:
    owner = (lock_dir / _LOCK_OWNER_NAME).read_text(encoding="utf-8")
    return int(owner.strip()) == os.getpid()
  except (OSError, ValueError):
    return False


_LOCK_HEARTBEAT_NAME = "heartbeat"


def _last_heartbeat(lock_dir: Path) -> int | None:
  try:
    return (lock_dir / _LOCK_HEARTBEAT_NAME).stat().st_mtime_ns
  except OSError:
    return None


@contextmanager
def lock_heartbeat(lock_dir: Path, interval_s: float) -> Generator[None, None, None]:
  """Touch a file inside the held `lock_dir` every `interval_s`, from a thread.

  The beat proves that the holder's process is alive and still inside the step,
  not that it is making progress: a holder hung in a call that never returns
  keeps beating until its process ends. Waiters give up only once the beat
  stops changing.
  """
  stop = threading.Event()
  beat = lock_dir / _LOCK_HEARTBEAT_NAME

  def run() -> None:
    while not stop.is_set():
      with suppress(OSError):
        beat.touch()
      stop.wait(interval_s)

  thread = threading.Thread(target=run, name="birdnet-lock-heartbeat", daemon=True)
  thread.start()
  try:
    yield
  finally:
    stop.set()
    thread.join()
