"""No ``multiprocessing`` Event or Condition is reachable from a running session.

Both run every operation inside a lock that has no owner, so a process killed
inside one wedges the next caller in every other process, and ``set()`` waits
for every registered waiter, forever for a dead one. The pipeline's signals are
flags, finish signals and semaphores instead (see ``SharedFlag`` and
``FinishSignal``). The protocols the children are typed with cannot tell a
``multiprocessing.Event`` from a flag, so this walks the session's objects and
what each child process was started with, and fails on the next ``ctx.Event()``.
"""

from __future__ import annotations

import functools
import multiprocessing.synchronize
from collections.abc import Iterator
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import BuiltinMethodType, MethodType
from typing import Any

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import fake_predict_session

pytestmark = pytest.mark.no_tf

_BUILT_ON_A_CONDITION = (
  multiprocessing.synchronize.Event,
  multiprocessing.synchronize.Condition,
)
# Objects of these packages are opened up; anything else is a leaf. The
# pipeline's signals live in birdnet objects, multiprocessing processes and
# queues, threads and plain containers.
_WALKED_PACKAGES = ("birdnet", "multiprocessing", "threading")


def _members(obj: object) -> Iterator[tuple[str, Any]]:
  if isinstance(obj, functools.partial):
    # Its fields are neither in __dict__ nor in __slots__.
    yield ".func", obj.func
    yield ".args", obj.args
    yield ".keywords", obj.keywords
    return
  if isinstance(obj, dict):
    for key, value in obj.items():
      yield f"[{key!r}]", value
    return
  if isinstance(obj, (list, tuple, set, frozenset)):
    for i, value in enumerate(obj):
      yield f"[{i}]", value
    return
  if isinstance(obj, (MethodType, BuiltinMethodType)):
    yield ".__self__", obj.__self__
    return
  if isinstance(obj, (type, str, bytes, Path, np.ndarray, np.generic)):
    return
  if not type(obj).__module__.startswith(_WALKED_PACKAGES):
    return
  for cls in type(obj).__mro__:
    for name in getattr(cls, "__slots__", ()):
      if hasattr(obj, name):
        yield f".{name}", getattr(obj, name)
  for name, value in getattr(obj, "__dict__", {}).items():
    yield f".{name}", value


def _built_on_a_condition(roots: dict[str, object]) -> list[str]:
  found: list[str] = []
  seen: set[int] = set()
  stack: list[tuple[str, object]] = list(roots.items())
  while stack:
    path, obj = stack.pop()
    if id(obj) in seen:
      continue
    seen.add(id(obj))
    if isinstance(obj, _BUILT_ON_A_CONDITION):
      found.append(f"{path}: {type(obj).__name__}")
      continue
    stack.extend((path + name, member) for name, member in _members(obj))
  return sorted(found)


def test_nothing_a_session_shares_is_built_on_a_condition(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # start() drops the process's target and arguments, so a signal that only a
  # child holds would be out of the walk's reach without this record.
  started: list[tuple[object, ...]] = []
  start = BaseProcess.start

  def recording_start(self: BaseProcess) -> None:
    started.append((self._target, self._args, self._kwargs))  # type: ignore[attr-defined]
    start(self)

  monkeypatch.setattr(BaseProcess, "start", recording_start)

  with fake_predict_session(
    tmp_path,
    n_workers=2,
    n_producers=2,
    show_stats="progress",
    progress_callback=lambda stats: None,
    on_file_complete=lambda result: None,
  ) as session:
    res = session._resources
    # Or the walk could pass on a session that never built these.
    assert res.stats_resources.perf_res_finish_signal is not None
    assert res.file_completion_resources.enabled
    assert session._process_manager._perf_tracker_process is not None
    # Two producers, two workers and the tracker.
    assert len(started) == 5, started

    found = _built_on_a_condition({"session": session, "started": started})

  assert not found, "\n".join(found)
