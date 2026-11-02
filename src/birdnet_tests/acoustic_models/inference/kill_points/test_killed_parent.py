"""A pipeline process whose parent died must exit, not wait for it.

The producers, workers and performance tracker leave their idle poll only on
the cancel or end event, and only the parent sets those. ``daemon=True`` does
not help: it acts when the parent's interpreter exits normally, not when the
parent is killed (``kill -9``, the OOM killer, a test runner's timeout kill),
crashes, or leaves through ``os._exit``. Each child would then wait in its
one-second polls for a signal nobody will send, holding its model and, on a
GPU, its device memory, until someone kills it by hand. So every poll also
asks whether the parent is still there, see ``parent_is_alive``.

The parent under test is a top-level process of its own, this file run as a
script, rather than a child of the test process: that gives it its own
resource tracker, as any script or test runner has, and lets the test kill it
the way the OOM killer would and watch its children from the outside. It
reports its children over stdout; where the kill has to land mid-run, the
report goes out from the progress callback once some segments are processed,
so the test knows a run was in flight and work was left.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import suppress
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal

import psutil
import pytest

import birdnet_tests
from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet_tests.fake_acoustic_backend import fake_predict_session, write_marked_audio

pytestmark = pytest.mark.no_tf

# Bounds a wedge, not a duration: a child notices within its 1 s poll and is
# gone a couple of seconds later, its log drain included.
_EXIT_DEADLINE_S = 30.0
# How long the parent gets to start an interpreter, a session and a run: a few
# seconds here, longer on a loaded three-core runner.
_START_DEADLINE_S = 120.0
# Long enough that the workers are unambiguously inside a batch when the kill
# lands, see test_killed_children.
_STALL_S = 0.5
# One producer, two workers and the performance tracker.
_N_CHILDREN = 4
# Two workers at one segment per batch: the first progress report leaves most
# of the run outstanding.
_N_SEGMENTS = 8

_KillWhen = Literal[
  "idle",
  "mid-run",
  "waiting-on-producer",
  "waiting-on-worker",
  "waiting-on-tracker",
  "holding-cancel-lock",
  "after-forking",
]
# Killed with a run under way; the others between runs.
_IN_FLIGHT = (
  "mid-run",
  "waiting-on-producer",
  "waiting-on-worker",
  "waiting-on-tracker",
)

_REPORT_PREFIX = "REPORT "


class _Reporter:
  """Writes the parent's one report line; the first caller wins."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._done = False

  def __call__(self, **fields: object) -> None:
    with self._lock:
      if self._done:
        return
      self._done = True
    sys.stdout.write(_REPORT_PREFIX + json.dumps(fields) + "\n")
    sys.stdout.flush()


def _parent_main(tmp_dir: str, kill_when: _KillWhen) -> None:
  """The process that gets killed: it starts a session and names its children.

  It never leaves the session on its own; the test takes it down from outside.
  """
  from multiprocessing import resource_tracker

  tmp_path = Path(tmp_dir)
  files = [str(write_marked_audio(tmp_path / "in.wav", n_segments=_N_SEGMENTS))]
  report = _Reporter()
  in_flight = kill_when in _IN_FLIGHT
  holder: BaseProcess | None = None

  def describe(stats: AcousticProgressStats | None) -> None:
    report(
      children=sorted(p.pid for p in mp.active_children() if p is not holder),
      holder=None if holder is None else holder.pid,
      # None where multiprocessing runs no resource tracker (Windows).
      tracker=getattr(resource_tracker._resource_tracker, "_pid", None),
      processed=None if stats is None else stats.processed_segments,
      total=None if stats is None else stats.total_segments,
    )

  def on_progress(stats: AcousticProgressStats) -> None:
    if in_flight and stats.processed_segments > 0:
      describe(stats)

  with fake_predict_session(
    tmp_path,
    n_workers=2,
    seconds_per_batch=_STALL_S if in_flight else 0.0,
    show_stats="progress",
    progress_callback=on_progress,
  ) as session:
    res = session._process_manager._res
    if not in_flight:
      session.run(files)
      if kill_when == "after-forking":
        # What a fork-mode pool or an os.fork() in the application leaves
        # behind: a process holding the write ends of the children's parent
        # sentinels past this process's death.
        holder = mp.get_context("fork").Process(target=time.sleep, args=(3600,))
        holder.start()
      elif kill_when == "holding-cancel-lock":
        # A parent killed inside is_set()/set() on the cancel event leaves its
        # lock held; a child that asked the event before the parent would
        # block for good.
        res.processing_resources.cancel_event._cond.acquire()
      describe(None)
      time.sleep(3600)
    threading.Thread(target=session.run, args=(files,), daemon=True).start()
    if kill_when == "mid-run":
      time.sleep(3600)
    # The main thread parked inside a child's finish_signal.wait(): the kill
    # then lands on a registered waiter, and that child's own set() would never
    # return without the guard after its run (see ChildSideEvent).
    finish_signal = {
      "waiting-on-producer": res.producer_resources.finish_signals[0],
      "waiting-on-worker": res.worker_resources.finish_signals[0],
      "waiting-on-tracker": res.stats_resources.perf_res_finish_signal,
    }[kill_when]
    assert finish_signal is not None
    finish_signal.wait()
    time.sleep(3600)


def _start_parent(
  tmp_path: Path, kill_when: str
) -> tuple[subprocess.Popen[str], queue.Queue[str | None]]:
  env = dict(os.environ)
  # The test package is importable here through pytest, not in a fresh interpreter.
  env["PYTHONPATH"] = os.pathsep.join(
    p
    for p in (str(Path(birdnet_tests.__file__).parents[1]), env.get("PYTHONPATH"))
    if p
  )
  parent = subprocess.Popen(
    [sys.executable, __file__, str(tmp_path), kill_when],
    stdout=subprocess.PIPE,
    text=True,
    env=env,
  )
  lines: queue.Queue[str | None] = queue.Queue()

  def pump() -> None:
    assert parent.stdout is not None
    for line in parent.stdout:
      lines.put(line)
    lines.put(None)

  # Drains past the report too, so the parent never blocks on a full pipe.
  threading.Thread(target=pump, name="parent-stdout", daemon=True).start()
  return parent, lines


def _read_report(
  lines: queue.Queue[str | None], parent: subprocess.Popen[str]
) -> dict[str, Any]:
  deadline = time.monotonic() + _START_DEADLINE_S
  while True:
    try:
      line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
    except queue.Empty:
      pytest.fail("the parent did not report its children in time")
    if line is None:
      pytest.fail(f"the parent exited (code {parent.wait()}) before reporting")
    if line.startswith(_REPORT_PREFIX):
      return json.loads(line[len(_REPORT_PREFIX) :])


def _still_running(proc: psutil.Process) -> bool:
  # A zombie has exited; it only waits for whoever inherited it to reap it.
  try:
    return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
  except psutil.NoSuchProcess:
    return False


def _wait_until_gone(procs: list[psutil.Process], deadline: float) -> list[int]:
  """Poll until every process is gone or the deadline passes; return survivors."""
  while time.monotonic() < deadline and any(_still_running(p) for p in procs):
    time.sleep(0.1)
  return [p.pid for p in procs if _still_running(p)]


def _kill_parent_and_expect_children_to_follow(tmp_path: Path, kill_when: str) -> None:
  parent, lines = _start_parent(tmp_path, kill_when)
  # Everything the parent started, for the cleanup: once it is dead, its
  # children are nobody's.
  descendants: list[psutil.Process] = []
  try:
    info = _read_report(lines, parent)
    children = [psutil.Process(pid) for pid in info["children"]]
    # Or the assertions below would pass on a pipeline that never started.
    assert len(children) == _N_CHILDREN, info
    tracker = None if info["tracker"] is None else psutil.Process(info["tracker"])
    descendants = psutil.Process(parent.pid).children(recursive=True)
    assert {p.pid for p in children} <= {p.pid for p in descendants}, info
    if kill_when in _IN_FLIGHT:
      # Reported from the progress callback: a run under way, with work left.
      assert 0 < info["processed"] < info["total"], info
    # Alive when the kill lands, or an exit for another reason would pass.
    assert all(_still_running(p) for p in children), info

    parent.kill()
    parent.wait(timeout=_EXIT_DEADLINE_S)

    deadline = time.monotonic() + _EXIT_DEADLINE_S
    survivors = _wait_until_gone(children, deadline)
    assert not survivors, (
      f"pipeline processes {survivors} outlived their killed parent by "
      f"{_EXIT_DEADLINE_S:.0f} s"
    )
    if info["holder"] is not None:
      # It holds the tracker's pipe as well; the tracker is judged without it.
      with suppress(psutil.NoSuchProcess):
        psutil.Process(info["holder"]).kill()
    if tracker is not None:
      # The children held its pipe open; with them gone it unlinks what the
      # parent registered and leaves too.
      assert not _wait_until_gone([tracker], deadline), (
        "the resource tracker outlived the pipeline processes"
      )
  finally:
    if parent.poll() is None:
      # Still alive, so the test failed before the kill: collect what it
      # started since the report, or before one.
      with suppress(psutil.NoSuchProcess):
        descendants = psutil.Process(parent.pid).children(recursive=True)
      parent.kill()
      parent.wait()
    for proc in descendants:
      with suppress(psutil.NoSuchProcess):
        proc.kill()


def test_idle_children_exit_when_the_parent_is_killed(tmp_path: Path) -> None:
  """Between runs every child sits in its start-signal poll."""
  _kill_parent_and_expect_children_to_follow(tmp_path, "idle")


def test_busy_children_exit_when_the_parent_is_killed(tmp_path: Path) -> None:
  """Mid-run the children are inside inference and audio decoding, not a poll."""
  _kill_parent_and_expect_children_to_follow(tmp_path, "mid-run")


@pytest.mark.parametrize("waiter", ["producer", "worker", "tracker"])
def test_a_child_whose_parent_died_waiting_on_its_finish_signal_exits(
  tmp_path: Path, waiter: str
) -> None:
  """A dead waiter makes ``Event.set()`` block for good; the child must skip it."""
  _kill_parent_and_expect_children_to_follow(tmp_path, f"waiting-on-{waiter}")


def test_children_exit_when_the_parent_died_holding_the_cancel_lock(
  tmp_path: Path,
) -> None:
  """The parent is asked before the cancel event, whose lock a dead parent may hold."""
  _kill_parent_and_expect_children_to_follow(tmp_path, "holding-cancel-lock")


@pytest.mark.skipif(sys.platform == "win32", reason="fork is POSIX-only")
def test_children_exit_although_a_forked_process_holds_their_sentinels(
  tmp_path: Path,
) -> None:
  """A forked holder keeps the sentinel pipe open; reparenting still shows the death."""
  _kill_parent_and_expect_children_to_follow(tmp_path, "after-forking")


if __name__ == "__main__":
  _parent_main(sys.argv[1], sys.argv[2])
