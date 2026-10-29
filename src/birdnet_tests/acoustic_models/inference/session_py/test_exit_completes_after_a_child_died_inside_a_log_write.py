"""Leaving a session completes promptly when the session log is wedged.

Every child's log records cross one `multiprocessing.Queue`, whose feeder
thread holds the queue's write lock around the pipe write on POSIX. A child
killed inside that write leaves the lock held and, when the write was cut
part-way, a half-written record in the pipe: every other child's feeder then
waits for the lock, which `multiprocessing` joins when that child exits, and
the parent's log writer blocks on the record's remainder for good. The
survivors must give their feeder up after a bounded drain and teardown must
give the writer up after a bounded wait, instead of spending the whole
teardown budget on each.

The first case kills the worker inside such a write: the parent's writer is
held back during the run so that the pipe is full and the worker's feeder
sits inside a write with the lock; whether that write was cut part-way depends
on the pipe's page layout, so the writer's bound is pinned by the second case,
where the writer never reads at all. Windows has no write lock and a
message-mode pipe, so there only the second case costs anything.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import birdnet.acoustic.inference.file_writer as fw_mod
from birdnet_tests.fake_acoustic_backend import (
  LogBurstBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_DEADLINE_S = 20.0

CASES = {
  "a_worker_killed_inside_a_log_write": True,
  "a_writer_that_never_reads": False,
}


@pytest.mark.parametrize("kill_worker", CASES.values(), ids=CASES.keys())
def test_exit_returns_with_the_log_wedged(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill_worker: bool
) -> None:
  resume = threading.Event()
  original_loop = fw_mod.QueueFileWriter._run_main_loop

  def held_back_loop(self: fw_mod.QueueFileWriter) -> None:
    resume.wait()
    original_loop(self)

  monkeypatch.setattr(fw_mod.QueueFileWriter, "_run_main_loop", held_back_loop)

  audio = write_marked_audio(tmp_path / "in.wav", 4)
  session = fake_predict_session(tmp_path, n_workers=1, backend=LogBurstBackend)
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  workers = manager._worker_processes  # noqa: SLF001
  producers = manager._producer_processes  # noqa: SLF001
  assert workers and producers
  done = threading.Event()
  errors: list[BaseException] = []
  try:
    session.run([str(audio)])
    time.sleep(0.2)
    if kill_worker:
      workers[0].kill()
      resume.set()

    def leave() -> None:
      try:
        session.__exit__(None, None, None)
      except BaseException as e:  # noqa: BLE001
        errors.append(e)
      finally:
        done.set()

    threading.Thread(target=leave, name="session-exit", daemon=True).start()
    assert done.wait(_DEADLINE_S), (
      f"__exit__ did not return within {_DEADLINE_S:.0f} s with the session log "
      f"wedged; the teardown waited out a log write that never completes"
    )
    assert errors == []
    for child in [*producers, *workers]:
      assert not child.is_alive(), child.name
  finally:
    if not done.is_set():
      for child in [*producers, *workers]:
        child.kill()
    # The writer of the second case is parked for good; let it go.
    resume.set()
