"""A deadline hands out what is left of its budget and never goes negative."""

from __future__ import annotations

import time

import pytest

from birdnet.acoustic.inference.core.sync import Deadline

pytestmark = pytest.mark.no_tf


def test_remaining_shrinks_with_time_and_stops_at_zero() -> None:
  deadline = Deadline(0.05)
  first = deadline.remaining()
  assert 0 < first <= 0.05
  time.sleep(0.07)
  assert deadline.remaining() == 0.0
  assert deadline.budget_s == 0.05


def test_remaining_is_the_whole_budget_until_the_clock_moves(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Windows before 3.13 reads time.monotonic() in 15.6 ms ticks, so the first
  # call usually sees the construction-time value; at this one, end minus now
  # rounds to a hair above the budget.
  monkeypatch.setattr(time, "monotonic", lambda: 127.953125)
  assert Deadline(0.05).remaining() == 0.05
