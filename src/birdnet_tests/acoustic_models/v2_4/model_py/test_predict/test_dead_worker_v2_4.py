"""A worker dying mid-run must fail the run, not hang it forever.

The parent waits for one sentinel per worker on the results queue and for each
child's finish signal, both without a timeout, because a healthy run takes as
long as the audio requires. A worker killed from the outside -- the OOM killer,
or a native crash -- sends neither, so before the liveness check the parent
waited forever with no output at all: the user saw birdnet freeze with no way
to tell a dead worker from a working one.

Killing the worker with SIGKILL is exactly what the OOM killer does, so this
reproduces the real failure rather than a simulation of it. Two kills are
covered: one while the worker is parked on its start signal, which pins the
diagnosis deterministically because the worker then holds nothing, and one
mid-batch, which is the realistic memory-pressure case and is where a killed
worker can leave a lock held (#73) or a half-written message in a queue (#77).
"""

import faulthandler
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from birdnet.globals import READING_FLAG
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG

# Generous upper bound: without the liveness check this never returns at all.
# The check reports within one poll interval once the results queue goes quiet.
_FAILURE_DEADLINE_S = 120.0
# TEST_FILE_LONG is 120 s and the model's segment is 3 s.
_EXPECTED_SEGMENTS = 40


def _copies(src: Path, tmp_path: Path, n: int) -> list[str]:
  # validate_input_files de-duplicates by absolute path, so the same file
  # passed n times collapses to one input and the run ends too quickly.
  return [
    str(shutil.copyfile(src, tmp_path / f"copy_{i}{src.suffix}")) for i in range(n)
  ]


def test_killed_worker_fails_the_run_with_a_diagnosis(tmp_path: Path) -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  files = _copies(Path(TEST_FILE_LONG), tmp_path, 4)

  outcome: dict = {}
  finished = threading.Event()
  killed = threading.Event()

  def run() -> None:
    # The whole session lives in this thread: if the run wedges, the `with`
    # must not be exited from another thread, or teardown blocks on the wedged
    # pipeline and the deadline assertion below never gets to report.
    try:
      with model.predict_session(n_workers=2, top_k=None) as session:
        workers = session._process_manager._worker_processes
        assert workers is not None
        # Killed while parked on its start signal, before any work is handed
        # out, so it holds nothing: whatever the diagnosis says here is about
        # the liveness check alone. The mid-batch kill, where the worker can
        # die holding a lock or half-way through a queue message, is covered
        # by the test below.
        workers[0].kill()
        workers[0].join(timeout=30)
        assert not workers[0].is_alive(), "the worker did not actually die"
        killed.set()

        session.run(files)
    except BaseException as e:  # noqa: BLE001 - recorded and asserted on below
      outcome["error"] = e
    finally:
      finished.set()

  runner = threading.Thread(target=run, name="dead-worker-run", daemon=True)
  runner.start()

  assert finished.wait(timeout=_FAILURE_DEADLINE_S), (
    f"run() did not return within {_FAILURE_DEADLINE_S:.0f} s with a dead "
    f"worker; the parent is waiting on a child that no longer exists."
  )
  assert killed.is_set(), "the worker was never killed, so nothing was tested"

  error = outcome.get("error")
  assert error is not None, "run() returned successfully despite a killed worker"
  assert isinstance(error, RuntimeError), f"unexpected error type: {error!r}"

  # The point of the check is the diagnosis, not merely that something failed:
  # every cancellation yields a RuntimeError, including a progress_callback
  # that raises. Only a detected child death names the process and exit code.
  message = str(error)
  assert "exited unexpectedly" in message, (
    f"error does not identify a dead child, so this could be any "
    f"cancellation: {message!r}"
  )
  assert re.search(r"Worker|worker", message), f"process not named: {message!r}"
  assert isinstance(error.__cause__, ChildProcessError), (
    f"the ChildProcessError should be chained as the cause, got {error.__cause__!r}"
  )


def test_worker_killed_mid_batch_fails_the_run_and_tears_down(
  tmp_path: Path,
) -> None:
  """The realistic OOM path: killed while it is doing work, not while parked.

  A worker killed here can be anywhere -- scanning the ring under the shared
  lock, updating a counter, or half-way through writing a result to its queue.
  None of those recover on their own: the lock is a semaphore nobody will post
  again, and a truncated queue message stops a reader for good with nothing
  raised.

  Read this as a broad regression test against wedging, not as the proof of
  either fix. What it pins is that a mid-run SIGKILL still ends in a returned
  RuntimeError; it deliberately does not assert *how long* teardown took,
  because the pre-existing grace period would terminate a wedged survivor after
  30 s and let this pass either way. The fix for the leaked lock is pinned
  deterministically, on exit codes, in test_leaked_ring_lock_v2_4.py, and the
  diagnosis is pinned by the parked-kill test above.

  Two workers on purpose: with one, the killed worker leaves no survivors, and
  the lock it may be holding has nobody left to block. Which of the two ends up
  holding what at kill time is not controllable from here.

  The kill is triggered by a slot going READING, not by a progress callback:
  that happens on the first batch rather than on the first stats interval a
  second later. A fast runner finishes this much audio inside that second, and
  killing a worker that has already signalled it is done proves nothing -- it
  is not an error, so the run rightly succeeded and the test failed.
  """
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  files = _copies(Path(TEST_FILE_LONG), tmp_path, 4)

  outcome: dict = {}
  finished = threading.Event()
  killed = threading.Event()
  started = threading.Event()

  def kill_a_worker_once_the_pipeline_is_working() -> None:
    assert started.wait(timeout=_FAILURE_DEADLINE_S)
    ring = outcome["ring"]
    flags = ring.rf_flags.get_array(ring._rf_flags_memory)
    finish_signals = outcome["finish_signals"]
    workers = outcome["workers"]
    deadline = time.monotonic() + _FAILURE_DEADLINE_S
    while time.monotonic() < deadline:
      if finish_signals[0].is_set():
        # It got through all its work first; killing it now would be the
        # ordinary end-of-run shutdown, not a death.
        return
      if any(flag == READING_FLAG for flag in flags):
        workers[0].kill()
        killed.set()
        return
      time.sleep(0.01)

  def run() -> None:
    # As above: the session must be entered and exited from one thread only.
    try:
      with model.predict_session(n_workers=2, top_k=None) as session:
        resources = session._resources
        outcome["workers"] = list(session._process_manager._worker_processes or [])
        outcome["finish_signals"] = resources.worker_resources.finish_signals
        outcome["ring"] = resources.ring_buffer_resources
        started.set()
        session.run(files)
    except BaseException as e:  # noqa: BLE001 - recorded and asserted on below
      outcome["error"] = e
    finally:
      finished.set()

  killer = threading.Thread(
    target=kill_a_worker_once_the_pipeline_is_working,
    name="dead-worker-killer",
    daemon=True,
  )
  runner = threading.Thread(target=run, name="dead-worker-midbatch", daemon=True)
  killer.start()
  runner.start()

  if not finished.wait(timeout=_FAILURE_DEADLINE_S):
    # The stacks are the only usable evidence for a wedge like this, and a
    # hanging daemon thread produces no traceback of its own.
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    raise AssertionError(
      f"the run did not finish within {_FAILURE_DEADLINE_S:.0f} s after a "
      f"worker was killed mid-batch; thread stacks dumped above"
    )

  assert killed.is_set(), (
    "no worker was ever killed while the pipeline had work in flight, so "
    "nothing was tested"
  )
  error = outcome.get("error")
  assert isinstance(error, RuntimeError), (
    f"the run must fail rather than return a partial result, got {error!r}"
  )


def test_healthy_run_is_unaffected_and_the_check_polls_rather_than_per_block(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A healthy run must be undisturbed, and the check must stay a poll.

  It is reached only from a wait that has already timed out after a full
  second, so its cost scales with *idle* time, never with the work done. A
  healthy run does idle: on a session's first run the consumer polls while the
  workers are still loading their models. What must never happen is the check
  becoming per-block -- this file yields 40 segments, so a per-block check
  would fire at least 40 times regardless of duration.

  Bounding it by the elapsed seconds pins exactly that, and unlike a fixed
  count it cannot go stale when pipeline timing shifts. Asserting merely that
  the run succeeds would pass even if the check had never been wired up.
  """
  from birdnet.acoustic.inference.process_manager import ProcessManager

  calls: list[int] = []
  original = ProcessManager.raise_if_child_died

  def counting(self: ProcessManager) -> None:
    calls.append(1)
    original(self)

  monkeypatch.setattr(ProcessManager, "raise_if_child_died", counting)

  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  target = _copies(Path(TEST_FILE_LONG), tmp_path, 1)

  with model.predict_session(n_workers=2, top_k=None) as session:
    start = time.perf_counter()
    result = session.run(target)
    elapsed_s = time.perf_counter() - start

  assert result is not None
  assert len(result.unprocessable_inputs) == 0
  assert result.species_probs.shape[1] == _EXPECTED_SEGMENTS

  # Lower bound first: without it every assertion here would still pass with
  # the check deleted from the pipeline entirely. A session's first run always
  # idles while the workers load their models, so it must fire at least once.
  assert calls, (
    "the liveness check never ran, so it is not wired into the pipeline at all"
  )

  # One waiter can fire at most once per second; a couple of waits overlap
  # (consumer, then the tail signals), so allow a small constant on top.
  budget = int(elapsed_s) + 4
  assert len(calls) <= budget, (
    f"liveness check ran {len(calls)} times in {elapsed_s:.1f} s over "
    f"{_EXPECTED_SEGMENTS} segments; it must poll idle time, not run per "
    f"block (budget {budget})"
  )
