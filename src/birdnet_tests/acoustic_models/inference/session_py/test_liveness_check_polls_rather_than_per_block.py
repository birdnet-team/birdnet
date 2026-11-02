"""A healthy run must be undisturbed, and the liveness check must stay a poll.

`ProcessManager.raise_if_child_died` is reached only from a wait that has
already timed out after a full second, so its cost scales with *idle* time,
never with the work done. A healthy run does idle: on a session's first run
the consumer polls while the workers are still loading their models, which
the slow-loading backend reproduces. What must never happen is the check
becoming per-block: a per-block check fires once per segment regardless of
duration.

Bounding it by the elapsed seconds pins exactly that, and unlike a fixed
count it cannot go stale when pipeline timing shifts. Asserting merely that
the run succeeds would pass even if the check had never been wired up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager
from birdnet_tests.fake_acoustic_backend import (
  SlowLoadingBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

# Sixteen segments at 0.2 s on one worker is about three seconds of work, which
# arrives block by block well inside the consumer's one-second poll, so no poll
# fires once the run is under way -- while a check per segment would add sixteen
# calls, far past the budget below.
_N_SEGMENTS = 16
_STALL_S = 0.2


def test_healthy_run_is_unaffected_and_the_check_polls_rather_than_per_block(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  calls: list[int] = []
  original = ProcessManager.raise_if_child_died

  def counting(self: ProcessManager) -> None:
    calls.append(1)
    original(self)

  monkeypatch.setattr(ProcessManager, "raise_if_child_died", counting)

  target = str(write_marked_audio(tmp_path / "in.wav", _N_SEGMENTS))
  with fake_predict_session(
    tmp_path, n_workers=1, seconds_per_batch=_STALL_S, backend=SlowLoadingBackend
  ) as session:
    start = time.perf_counter()
    result = session.run(target)
    elapsed_s = time.perf_counter() - start

  assert len(result.unprocessable_inputs) == 0
  assert result.species_probs.shape[1] == _N_SEGMENTS

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
    f"{_N_SEGMENTS} segments; it must poll idle time, not run per block "
    f"(budget {budget})"
  )
