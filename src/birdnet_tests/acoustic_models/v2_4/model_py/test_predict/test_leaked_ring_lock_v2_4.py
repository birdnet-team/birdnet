"""A ring-buffer lock nobody will release again must not wedge the workers.

`multiprocessing.Lock` is a POSIX semaphore. A worker killed while scanning the
ring for a readable slot -- SIGKILL from the OOM killer, a native crash -- never
posts it again, so every surviving worker blocks on the next scan forever. The
parent's liveness check (#72) reports the child that died but cannot free the
survivors: they are alive, just stuck. Before the fix that ended in the parent
terminating them after the teardown grace period; now they notice the run was
cancelled and leave on their own (issue #73).

The lock is leaked here by taking it in the parent and never releasing it,
which is the same situation the workers see and, unlike killing a worker at
exactly the right microsecond, is reproducible.
"""

import shutil
import threading
import time
from pathlib import Path

from birdnet.globals import READABLE_FLAG
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG

_N_WORKERS = 2
# prefetch_ratio defaults to 1, so the ring holds two slots per worker.
_N_SLOTS = _N_WORKERS * 2
# Generous upper bound on the whole cancelled run plus teardown. What this test
# distinguishes is *how* the workers ended, not how fast -- see the exit codes.
_DEADLINE_S = 180.0
# The workers have to load their model before they can reach the ring scan, so
# this bounds a cold start, not the stall itself.
_STALL_DEADLINE_S = 120.0


def _copies(src: Path, tmp_path: Path, n: int) -> list[str]:
  # validate_input_files de-duplicates by absolute path, so the same file passed
  # n times collapses to one input.
  return [
    str(shutil.copyfile(src, tmp_path / f"copy_{i}{src.suffix}")) for i in range(n)
  ]


def _wait_until_every_worker_is_stuck_at_the_lock(resources) -> bool:  # noqa: ANN001
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
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  files = _copies(Path(TEST_FILE_LONG), tmp_path, 4)

  outcome: dict = {}
  finished = threading.Event()

  def run() -> None:
    # The session lives entirely in this thread: if anything wedges, the `with`
    # must not be exited from the test thread, or teardown blocks there and the
    # deadline assertion below never gets to report.
    try:
      with model.predict_session(n_workers=_N_WORKERS, top_k=None) as session:
        resources = session._resources
        # Stand-in for the worker that died holding it.
        resources.worker_resources.ring_access_lock.acquire()

        outcome["workers"] = list(session._process_manager._worker_processes or [])

        def cancel_once_stalled() -> None:
          outcome["stalled"] = _wait_until_every_worker_is_stuck_at_the_lock(resources)
          session.cancel()

        canceller = threading.Thread(
          target=cancel_once_stalled, name="leaked-lock-canceller", daemon=True
        )
        canceller.start()

        session.run(files)
    except BaseException as e:  # noqa: BLE001 - recorded and asserted on below
      outcome["error"] = e
    finally:
      finished.set()

  runner = threading.Thread(target=run, name="leaked-lock-run", daemon=True)
  runner.start()

  assert finished.wait(timeout=_DEADLINE_S), (
    f"the session did not finish within {_DEADLINE_S:.0f} s with a leaked ring "
    f"lock; the workers are blocked on a lock nobody will release."
  )

  assert outcome.get("stalled"), (
    "the workers never reached the ring scan, so the leaked lock was never exercised"
  )

  error = outcome.get("error")
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
