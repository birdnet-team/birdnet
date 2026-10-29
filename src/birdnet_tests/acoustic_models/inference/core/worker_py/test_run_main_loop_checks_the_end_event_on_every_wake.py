"""An idle worker released by its start signal still honours the end event.

`end()` releases every idle start signal so the waits return at once, so a
release is not by itself a run start: the loop must look at the end and
cancel events after every wake and return without calling `run_main`.
"""

from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

import pytest

from birdnet.acoustic.inference.core.worker import WorkerBase

pytestmark = pytest.mark.no_tf


def _idle_worker(run_main) -> SimpleNamespace:  # noqa: ANN001
  stub = SimpleNamespace(
    _start_signal=threading.Semaphore(0),
    _end_event=threading.Event(),
    _cancel_event=threading.Event(),
    _finish_signal=threading.Event(),
    _log=lambda *args: None,
    run_main=run_main,
  )
  stub._check_cancel_event = MethodType(WorkerBase._check_cancel_event, stub)
  stub._check_end_event = MethodType(WorkerBase._check_end_event, stub)
  return stub


def test_a_wake_with_the_end_event_set_returns_without_running() -> None:
  runs: list[int] = []
  worker = _idle_worker(lambda: runs.append(1))
  worker._start_signal.release()
  worker._end_event.set()

  WorkerBase.run_main_loop(worker)

  assert runs == []
  assert not worker._finish_signal.is_set()


def test_a_wake_with_the_cancel_event_set_returns_without_running() -> None:
  runs: list[int] = []
  worker = _idle_worker(lambda: runs.append(1))
  worker._start_signal.release()
  worker._cancel_event.set()

  WorkerBase.run_main_loop(worker)

  assert runs == []


def test_a_wake_without_either_event_runs_once_and_waits_again() -> None:
  worker = _idle_worker(lambda: None)
  runs: list[int] = []

  def run_main() -> None:
    # A loop that takes a released start signal as a run start without
    # looking at the end event would come straight back here; fail instead of
    # spinning.
    assert runs == [], "run_main called again after the session ended"
    runs.append(1)
    # End the session from inside the run and release the next wait at once,
    # as `end()` does, so the loop returns without waiting out a poll.
    worker._end_event.set()
    worker._start_signal.release()

  worker.run_main = run_main
  worker._start_signal.release()

  WorkerBase.run_main_loop(worker)

  assert runs == [1]
  assert worker._finish_signal.is_set()
