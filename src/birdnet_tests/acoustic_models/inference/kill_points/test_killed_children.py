"""Killing any child at any point must end the run, not wedge it.

These run on the fake backend, so the pipeline starts in a quarter of a second
and a kill point costs a test rather than a model download. That is the whole
reason they can be specific about *where* the child dies: the existing
end-to-end versions of these tests kill a worker at whatever moment a progress
callback or a ring-flag poll happens to catch, which is neither reproducible
nor aimed at anything in particular.

`seconds_per_batch` is what makes the aim possible. A worker that spends half a
second inside inference is inside inference when the test kills it, every time,
on every runner — no polling, no flag inspection, no race with a fast machine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from birdnet.acoustic.inference.session import AcousticSessionBase
from birdnet_tests.fake_acoustic_backend import (
  WideFakeAcousticBackend,
  fake_encode_session,
  fake_predict_session,
  write_marked_audio,
)

# The fake backend is pure numpy, so these exercise the pipeline on the
# TensorFlow-free surface too — the 3.14 lane otherwise runs almost none of it.
pytestmark = pytest.mark.no_tf

# Every run here is sub-second when healthy. This bounds a wedge, not a duration.
_DEADLINE_S = 60.0
# Long enough that a worker is unambiguously inside a batch when it is killed,
# short enough that the healthy part of the run stays quick.
_STALL_S = 0.5


def _corpus(tmp_path: Path, n_files: int = 2, n_segments: int = 4) -> list[str]:
  """Just enough work that the kill lands mid-run.

  Sized deliberately: with `seconds_per_batch` set, every extra segment is
  half a second of test. Eight segments across two workers is ~2 s of work
  against a kill at 0.5 s, which leaves plenty outstanding without paying for
  audio nobody looks at.
  """
  return [
    str(write_marked_audio(tmp_path / f"in_{i}.wav", n_segments))
    for i in range(n_files)
  ]


def _run_in_thread(
  session_factory: Callable[[], AcousticSessionBase],
  files: list[str],
  kill: Callable[[AcousticSessionBase], None],
) -> dict:
  """Run a session to completion in one thread, and report what happened.

  The session is entered and exited on the same thread on purpose: if anything
  wedges, exiting the `with` from the test thread would block there instead,
  and the deadline assertion would never be reached.
  """
  outcome: dict = {}
  finished = threading.Event()

  def run() -> None:
    try:
      with session_factory() as session:
        kill(session)
        session.run(files)
    except BaseException as e:  # noqa: BLE001 - asserted on by the caller
      outcome["error"] = e
    finally:
      finished.set()

  threading.Thread(target=run, name="kill-point-run", daemon=True).start()
  outcome["finished"] = finished.wait(timeout=_DEADLINE_S)
  return outcome


def _kill_after(
  process_getter: Callable[[AcousticSessionBase], BaseProcess], delay_s: float
) -> Callable[[AcousticSessionBase], None]:
  """Kill the process this returns, `delay_s` into the run."""

  def kill(session: AcousticSessionBase) -> None:
    target = process_getter(session)

    if delay_s <= 0:
      # Synchronously, before the run starts. Handing this to a thread would
      # race the run and could kill a worker that has already finished -- the
      # same mistake that made an earlier version of the end-to-end test pass
      # while proving nothing.
      target.kill()
      return

    def later() -> None:
      time.sleep(delay_s)
      target.kill()

    threading.Thread(target=later, name="killer", daemon=True).start()

  return kill


def _worker(index: int) -> Callable[[AcousticSessionBase], BaseProcess]:
  def get(session: AcousticSessionBase) -> BaseProcess:
    procs = session._process_manager._worker_processes
    assert procs is not None
    return procs[index]

  return get


def _producer(index: int) -> Callable[[AcousticSessionBase], BaseProcess]:
  def get(session: AcousticSessionBase) -> BaseProcess:
    procs = session._process_manager._producer_processes
    assert procs is not None
    return procs[index]

  return get


def _assert_reported(outcome: dict, what: str) -> None:
  assert outcome["finished"], (
    f"the run did not return within {_DEADLINE_S:.0f} s after {what} was "
    f"killed; the pipeline is wedged on a child that no longer exists"
  )
  error = outcome.get("error")
  assert isinstance(error, RuntimeError), (
    f"killing {what} must fail the run rather than return a partial result, "
    f"got {error!r}"
  )
  assert "exited unexpectedly" in str(error), (
    f"the failure must name the dead child; got {str(error)[:200]!r}"
  )


def test_worker_killed_while_inside_inference(tmp_path: Path) -> None:
  """The realistic OOM moment, aimed at rather than stumbled into.

  A worker killed here holds a claimed ring slot and may hold the ring lock,
  the slot-gauge lock, or a queue write lock. This is the case that hung five
  POSIX CI lanes before #82.
  """
  outcome = _run_in_thread(
    lambda: fake_predict_session(tmp_path, n_workers=2, seconds_per_batch=_STALL_S),
    _corpus(tmp_path),
    _kill_after(_worker(0), delay_s=_STALL_S),
  )
  _assert_reported(outcome, "a worker mid-inference")


def test_worker_killed_before_it_starts_working(tmp_path: Path) -> None:
  """Killed while parked on its start signal, holding nothing."""
  outcome = _run_in_thread(
    lambda: fake_predict_session(tmp_path, n_workers=2),
    _corpus(tmp_path),
    _kill_after(_worker(0), delay_s=0.0),
  )
  _assert_reported(outcome, "a parked worker")


def test_producer_killed_mid_run(tmp_path: Path) -> None:
  """Producers hold their own ring lock and the done-counter's lock."""
  outcome = _run_in_thread(
    lambda: fake_predict_session(tmp_path, n_workers=2, seconds_per_batch=_STALL_S),
    _corpus(tmp_path),
    _kill_after(_producer(0), delay_s=_STALL_S / 2),
  )
  _assert_reported(outcome, "a producer mid-run")


def test_worker_killed_during_encoding(tmp_path: Path) -> None:
  """Killed mid-encode, with blocks big enough to actually tear.

  The wide backend matters: an earlier version of this test claimed to cross
  the 16 KB framing threshold at `batch_size >= 4` while the default stub's
  16-wide embeddings kept every block near 300 bytes. With `emb_dim = 1024`
  one batch-4 block really is past 16 KB, the size at which POSIX splits a
  queue message into separate header and body writes -- where a killed writer
  can leave a header whose body never arrives.
  """
  outcome = _run_in_thread(
    lambda: fake_encode_session(
      tmp_path,
      n_workers=2,
      batch_size=4,
      seconds_per_batch=_STALL_S,
      backend=WideFakeAcousticBackend,
    ),
    _corpus(tmp_path),
    _kill_after(_worker(0), delay_s=_STALL_S),
  )
  _assert_reported(outcome, "a worker mid-encode")


def test_worker_killed_with_top_k_none_blocks(tmp_path: Path) -> None:
  """The configuration issue #83 names: ~45 KB result blocks.

  With `top_k=None` every block carries a score per species; at the wide
  backend's 6000 species that is well past every framing threshold, so a kill
  mid-run is a kill amid multi-write frames. The run must end with a
  diagnosis; before the consumer read through a sacrificial thread, a frame
  torn here could block the parent's main loop with nothing raised.
  """
  outcome = _run_in_thread(
    lambda: fake_predict_session(
      tmp_path,
      n_workers=2,
      top_k=None,
      seconds_per_batch=_STALL_S,
      backend=WideFakeAcousticBackend,
    ),
    _corpus(tmp_path),
    _kill_after(_worker(0), delay_s=_STALL_S),
  )
  _assert_reported(outcome, "a worker with top_k=None")


@pytest.mark.parametrize("n_workers", [1, 3])
def test_the_survivors_do_not_matter(tmp_path: Path, n_workers: int) -> None:
  """With one worker there are no survivors; with three there are two.

  Worth pinning both: a leaked lock needs somebody left to block on it, so the
  one-worker case exercises a different path through teardown than the many-
  worker case, and the original regression test only ever ran the first.
  """
  outcome = _run_in_thread(
    lambda: fake_predict_session(
      tmp_path, n_workers=n_workers, seconds_per_batch=_STALL_S
    ),
    _corpus(tmp_path),
    _kill_after(_worker(0), delay_s=_STALL_S),
  )
  _assert_reported(outcome, f"a worker of {n_workers}")
