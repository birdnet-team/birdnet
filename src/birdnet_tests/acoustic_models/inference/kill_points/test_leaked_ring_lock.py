"""A ring-buffer lock nobody will release again must not wedge the workers.

`multiprocessing.Lock` is a semaphore on every platform, and a semaphore has no
owner to abandon it to the next waiter. A worker killed while scanning the ring
for a readable slot -- SIGKILL from the OOM killer, a native crash -- never
posts it again, so every surviving worker blocks on the next scan forever. The
parent's liveness check reports the child that died but cannot free the
survivors: they are alive, just stuck. They have to notice the run was
cancelled and leave on their own (issue #73).

The lock is leaked here by taking it in the parent and never releasing it,
which is the same situation the workers see and, unlike killing a worker at
exactly the right microsecond, is reproducible. The cancellation is supplied by
the test rather than by the liveness check, so what is pinned is the workers'
half of the chain; the parent's half is the kill points in
test_killed_children.py.

Only `worker_resources.ring_access_lock` is leaked here. The producers' own
ring lock and their done-counter take the same give-up path but are covered
only by the unit tests for it, not end-to-end.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from birdnet.acoustic.inference.resources import PipelineResources
from birdnet.acoustic.inference.session import AcousticSessionBase
from birdnet.globals import READABLE_FLAG
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)
from birdnet_tests.helper import run_session_in_thread

pytestmark = pytest.mark.no_tf

_N_WORKERS = 2
# prefetch_ratio is 1, so the ring holds two slots per worker.
_N_SLOTS = _N_WORKERS * 2
# Generous upper bound on the whole cancelled run plus teardown. What this test
# distinguishes is *how* the workers ended, not how fast -- see the exit codes.
# A worker that does not give up is terminated after the 30 s teardown grace,
# and that outcome has to be reported as such, not as a wedge.
_DEADLINE_S = 120.0
# The fake loads instantly, so this bounds the stall itself.
_STALL_DEADLINE_S = 30.0


def _wait_until_every_worker_is_stuck_at_the_lock(resources: PipelineResources) -> bool:
  """True once every worker holds a wake-up permit but has claimed no slot.

  A worker takes a permit from `sem_filled_slots` and then scans the ring under
  the lock, marking the slot it claims READING. So when the ring is full, no
  slot is READING and exactly one permit per worker has been taken, every
  worker is inside the scan -- which is where the leaked lock stops them.
  """
  ring = resources.ring_buffer_resources
  flags = ring.rf_flags.get_array(ring._rf_flags_memory)
  deadline = time.monotonic() + _STALL_DEADLINE_S
  while time.monotonic() < deadline:
    ring_is_full = all(flag == READABLE_FLAG for flag in flags)
    permits_taken = _N_SLOTS - ring.sem_filled_slots.get_value()
    if ring_is_full and permits_taken == _N_WORKERS:
      return True
    time.sleep(0.05)
  return False


def test_a_leaked_ring_lock_lets_the_workers_exit_instead_of_wedging(
  tmp_path: Path,
) -> None:
  # Eight segments: the four slots fill with work still queued behind them.
  files = [str(write_marked_audio(tmp_path / f"in_{i}.wav", 4)) for i in range(2)]
  outcome: dict = {}

  def leak_the_lock_and_cancel_once_stalled(session: AcousticSessionBase) -> None:
    resources = session._resources
    # Stand-in for the worker that died holding it.
    resources.worker_resources.ring_access_lock.acquire()
    outcome["workers"] = list(session._process_manager._worker_processes or [])

    def cancel_once_stalled() -> None:
      outcome["stalled"] = _wait_until_every_worker_is_stuck_at_the_lock(resources)
      session.cancel()

    threading.Thread(
      target=cancel_once_stalled, name="leaked-lock-canceller", daemon=True
    ).start()

  run = run_session_in_thread(
    lambda: fake_predict_session(tmp_path, n_workers=_N_WORKERS),
    files,
    leak_the_lock_and_cancel_once_stalled,
    _DEADLINE_S,
  )

  assert run["finished"], (
    f"the session did not finish within {_DEADLINE_S:.0f} s with a leaked ring "
    f"lock; the workers are blocked on a lock nobody will release."
  )
  assert outcome.get("stalled"), (
    "the workers never reached the ring scan, so the leaked lock was never exercised"
  )
  error = run.get("error")
  assert isinstance(error, RuntimeError), f"unexpected outcome: {error!r}"

  workers = outcome["workers"]
  assert len(workers) == _N_WORKERS
  for worker in workers:
    # A worker that had to be terminated carries a non-zero exit code (-15 on
    # POSIX). Exit code 0 is the whole point: it gave up on the lock, saw the
    # cancellation and shut itself down.
    assert worker.exitcode == 0, (
      f"'{worker.name}' exited with {worker.exitcode}: it did not give up on "
      f"the leaked lock and had to be terminated by teardown"
    )
