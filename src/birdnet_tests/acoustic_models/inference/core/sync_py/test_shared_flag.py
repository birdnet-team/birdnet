"""A process killed inside a flag operation leaves nothing the others block on.

``multiprocessing.Event`` runs ``is_set()``, ``set()`` and ``clear()`` inside a
lock that has no owner, so a process killed in any of them wedges the next
caller in every other process. Each round here kills a child spinning in one
operation at an arbitrary instant, then asks the parent's ``is_set()``,
``set()`` and ``clear()`` to return. The children use the pipeline's start
method, so a run under ``BIRDNET_START_METHOD`` shares the flag the way a
session does.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
import random
import threading
import time
from collections.abc import Callable

import pytest

from birdnet.acoustic.inference.core.sync import SharedFlag
from birdnet.core.start_method import resolve_start_method

pytestmark = pytest.mark.no_tf

# The same rounds against a ``ctx.Event()`` wedged 70 of 130 (is_set) and 100
# of 130 (set) on Linux under spawn, never fewer than 40 % in a batch, so this
# many rounds miss an Event-backed flag about once in 27 000 runs.
_ROUNDS = 20
# Bounds a wedge: each operation is a store or a load.
_DEADLINE_S = 10.0
# A spawned interpreter's start, longer on a loaded runner.
_START_S = 120.0


def _spin(
  flag: SharedFlag, op: str, spinning: multiprocessing.synchronize.Semaphore
) -> None:
  call = getattr(flag, op)
  for _ in range(1000):
    call()
  spinning.release()
  while True:
    call()


def _operations_return(flag: SharedFlag, deadline_s: float) -> bool:
  """Whether the parent's is_set, set and clear all return, and agree."""
  outcome: list[bool] = []

  def operate() -> None:
    flag.is_set()
    flag.set()
    was_set = flag.is_set()
    flag.clear()
    outcome.append(was_set and not flag.is_set())

  # A thread, so a wedged operation costs the deadline rather than the test.
  thread = threading.Thread(target=operate, name="flag-operations", daemon=True)
  thread.start()
  thread.join(timeout=deadline_s)
  if thread.is_alive():
    return False
  assert outcome == [True], "set() or clear() did not take"
  return True


def kill_rounds(
  make_flag: Callable[[mp.context.BaseContext], SharedFlag],
  op: str,
  rounds: int,
  deadline_s: float = _DEADLINE_S,
) -> int:
  """Kill a child spinning in ``op`` once per round; count the rounds that wedge."""
  ctx = mp.get_context(resolve_start_method())
  wedged = 0
  for _ in range(rounds):
    flag = make_flag(ctx)
    spinning = ctx.Semaphore(0)
    child = ctx.Process(target=_spin, args=(flag, op, spinning), daemon=True)
    child.start()
    try:
      assert spinning.acquire(timeout=_START_S), "the child never started spinning"
      time.sleep(random.uniform(0.0, 0.01))
    finally:
      child.kill()
      child.join()
    if not _operations_return(flag, deadline_s):
      wedged += 1
  return wedged


@pytest.mark.parametrize("op", ["is_set", "set"])
def test_a_child_killed_inside_a_flag_operation_leaves_the_parent_free(
  op: str,
) -> None:
  assert kill_rounds(SharedFlag, op, _ROUNDS) == 0


def test_a_flag_set_in_one_process_is_seen_in_another() -> None:
  ctx = mp.get_context(resolve_start_method())
  flag = SharedFlag(ctx)
  child = ctx.Process(target=flag.set)
  child.start()
  child.join(timeout=_START_S)

  assert child.exitcode == 0
  assert flag.is_set()
  flag.clear()
  assert not flag.is_set()
