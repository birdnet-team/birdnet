"""A child that dies without signalling must be reported, not waited on.

Every parent-side wait in a healthy run is unbounded on purpose. That is only
safe while the children are working: one killed by the OOM killer or crashing
in native code never sets its finish signal, so the parent would block forever
with no output. These tests pin the decision logic that turns that into an
error; the end-to-end proof is in
`acoustic_models/v2_4/model_py/test_predict/test_dead_worker_v2_4.py`.
"""

from __future__ import annotations

import threading

import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager


class _FakeProcess:
  def __init__(self, name: str, *, alive: bool, exitcode: int | None = None) -> None:
    self.name = name
    self.exitcode = exitcode
    self._alive = alive

  def is_alive(self) -> bool:
    return self._alive


class _StubManager:
  """Only what `raise_if_child_died` touches: the child/signal pairs."""

  def __init__(self, pairs: list[tuple[_FakeProcess, threading.Event]]) -> None:
    self._pairs = pairs

  def _iter_children_with_finish_signals(self):  # noqa: ANN202
    return iter(self._pairs)

  raise_if_child_died = ProcessManager.raise_if_child_died


def _signal(*, is_set: bool) -> threading.Event:
  event = threading.Event()
  if is_set:
    event.set()
  return event


def test_running_children_are_not_reported() -> None:
  manager = _StubManager(
    [
      (_FakeProcess("Worker-0", alive=True), _signal(is_set=False)),
      (_FakeProcess("Producer-0", alive=True), _signal(is_set=False)),
    ]
  )

  manager.raise_if_child_died()


def test_child_that_exited_after_signalling_is_not_reported() -> None:
  """The normal end-of-session shutdown: signalled, then exited."""
  manager = _StubManager(
    [(_FakeProcess("Worker-0", alive=False, exitcode=0), _signal(is_set=True))]
  )

  manager.raise_if_child_died()


def test_child_killed_without_signalling_is_reported() -> None:
  manager = _StubManager(
    [
      (_FakeProcess("Worker-0", alive=True), _signal(is_set=False)),
      # -9 is what the OOM killer leaves behind.
      (_FakeProcess("Worker-1", alive=False, exitcode=-9), _signal(is_set=False)),
    ]
  )

  with pytest.raises(ChildProcessError, match=r"Worker-1.*-9"):
    manager.raise_if_child_died()


def test_child_that_exited_zero_without_signalling_is_reported() -> None:
  """Exit code 0 is still wrong mid-run: nobody is left to finish the work."""
  manager = _StubManager(
    [(_FakeProcess("Producer-0", alive=False, exitcode=0), _signal(is_set=False))]
  )

  with pytest.raises(ChildProcessError, match="Producer-0"):
    manager.raise_if_child_died()


def test_no_children_is_not_an_error() -> None:
  _StubManager([]).raise_if_child_died()
