"""Available memory is the host's view, or the cgroup's when that is tighter."""

from __future__ import annotations

import pytest

import birdnet.acoustic.inference.memory as memory

pytestmark = pytest.mark.no_tf

_GIB = 2**30


def test_the_tighter_of_host_and_cgroup_wins(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(memory, "host_memory_available", lambda: 8 * _GIB)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: 3 * _GIB)
  assert memory.available_memory() == (3 * _GIB, "cgroup limit")
  assert memory.available_memory_bytes() == 3 * _GIB
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: 10 * _GIB)
  assert memory.available_memory() == (8 * _GIB, "host")
  assert memory.available_memory_bytes() == 8 * _GIB


def test_without_a_cgroup_limit_the_host_decides(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(memory, "host_memory_available", lambda: 8 * _GIB)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: None)
  assert memory.available_memory_bytes() == 8 * _GIB
