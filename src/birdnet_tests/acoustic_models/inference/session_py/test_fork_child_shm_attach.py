"""Under ``fork``, no pipeline child may attach the ring buffers itself.

``SharedMemory(name=..., create=False)`` calls
``multiprocessing.resource_tracker.register``, which takes a module-level
``threading.RLock`` that CPython does not reinitialize after ``fork``. A child
that inherits that lock held by a thread which does not exist in the child
blocks on the attach forever, with no timeout and nothing to interrupt it.

``fork`` children inherit the parent's mappings, so they never need to attach:
the parent does it before forking and the child uses what it inherited. This
pins that invariant by counting the attaches that happen in a process other than
the one that set the session up. It is deterministic -- the deadlock itself is a
race, so it is not what gets asserted here.

Both stats modes are covered because they start different sets of children:
``show_stats=None`` runs producers and workers only, while ``"progress"`` also
starts ``PerformanceTracker``, which attaches the ring flags of its own.
"""

import multiprocessing as mp
import os
from multiprocessing import resource_tracker
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import use_fork_or_skip

SAMPLE_RATE = 48_000
CLIP_DURATION_S = 9.0

_original_register = resource_tracker.register
_parent_pid: int | None = None
_registrations_path: Path | None = None


def _tracing_register(name: str, rtype: str) -> None:
  # Runs in the parent and in every fork child. Only the child copies matter,
  # and a child cannot report back through memory, so they append to a file.
  if os.getpid() != _parent_pid and rtype == "shared_memory":
    assert _registrations_path is not None
    with _registrations_path.open("a", encoding="utf-8") as f:
      f.write(f"{os.getpid()} {name}\n")
  _original_register(name, rtype)


@pytest.mark.fork
@pytest.mark.parametrize("show_stats", [None, "progress"])
def test_fork_children_do_not_attach_shared_memory_themselves(
  tmp_path: Path,
  show_stats: Literal["progress"] | None,
) -> None:
  use_fork_or_skip()
  assert mp.get_start_method(allow_none=True) == "fork"

  global _parent_pid, _registrations_path
  _parent_pid = os.getpid()
  _registrations_path = tmp_path / "shm_registrations.txt"

  rng = np.random.default_rng(0)
  audio = rng.standard_normal(int(SAMPLE_RATE * CLIP_DURATION_S)).astype(np.float32)

  resource_tracker.register = _tracing_register  # type: ignore[assignment]
  try:
    model = load("acoustic", "2.4", "tf", library="tflite")
    with model.predict_session(
      n_workers=1, top_k=None, show_stats=show_stats
    ) as session:
      session.run_arrays((audio * 0.1, SAMPLE_RATE))
  finally:
    resource_tracker.register = _original_register  # type: ignore[assignment]

  registrations = (
    _registrations_path.read_text(encoding="utf-8").splitlines()
    if _registrations_path.exists()
    else []
  )

  assert not registrations, (
    f"{len(registrations)} shared-memory attach(es) happened in a fork child "
    f"instead of being inherited from the parent. Each one calls "
    f"resource_tracker.register, whose lock is not reinitialized after fork and "
    f"can deadlock the child permanently: {registrations}"
  )
