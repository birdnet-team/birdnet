"""Leaving a session completes within one budget, whatever a child is doing.

A worker stuck in model loading never polls the end event, so a plain join on
it would wait for as long as the load takes: minutes for a slow network share,
forever for a hung interpreter. Teardown draws every wait from one deadline,
terminates what is still there when the kill reserve is all that is left, and
kills what ignores that.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import birdnet.acoustic.inference.process_manager as pm_mod
from birdnet_tests.fake_acoustic_backend import (
  FakeAcousticBackend,
  IgnoringTerminationBackend,
  NeverLoadingBackend,
  fake_model_path,
  fake_predict_session,
  loading_marker,
)

pytestmark = pytest.mark.no_tf

# Above the kill reserve, so the stuck worker gets a real grace period first.
BUDGET_S = 6.0
GRACE_S = BUDGET_S - pm_mod._KILL_RESERVE_S
# Terminating a child stuck in a Python sleep is immediate; this covers a slow
# runner, not a mechanism.
MARGIN_S = 5.0


@pytest.mark.parametrize(
  "backend",
  [NeverLoadingBackend, IgnoringTerminationBackend],
  ids=["dies_on_sigterm", "ignores_sigterm"],
)
def test_exit_with_a_worker_stuck_in_load_ends_within_the_budget(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: type[FakeAcousticBackend]
) -> None:
  monkeypatch.setattr(pm_mod, "TEARDOWN_BUDGET_S", BUDGET_S)
  session = fake_predict_session(tmp_path, n_workers=1, backend=backend)
  session.__enter__()
  assert session._process_manager is not None
  workers = session._process_manager._worker_processes
  assert workers is not None
  marker = loading_marker(fake_model_path(tmp_path))
  deadline = time.monotonic() + 20.0
  while not marker.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
  assert marker.exists(), "the worker never reached load(); nothing to test"
  assert workers[0].is_alive()

  done = threading.Event()
  took: dict[str, float] = {}

  def leave() -> None:
    t = time.perf_counter()
    try:
      session.__exit__(None, None, None)
    finally:
      took["s"] = time.perf_counter() - t
      done.set()

  threading.Thread(target=leave, name="teardown-budget-exit", daemon=True).start()
  finished = done.wait(timeout=BUDGET_S + MARGIN_S + 10)
  try:
    assert finished, "__exit__ did not return: teardown waited on the stuck worker"
    assert took["s"] < BUDGET_S + MARGIN_S, f"__exit__ took {took['s']:.1f} s"
    # The grace was honoured: a stuck child is not terminated right away.
    assert took["s"] >= GRACE_S - 0.2, f"__exit__ took only {took['s']:.1f} s"
    assert not workers[0].is_alive()
    # The thread joins behind the children kept their share: a healthy log
    # writer is joined, not declared wedged because the budget was spent.
    writers = [t for t in threading.enumerate() if t.name.endswith("-QueueFileWriter")]
    assert not [t for t in writers if t.is_alive()], "the log writer was abandoned"
  finally:
    for w in workers:
      if w.is_alive():
        w.kill()
