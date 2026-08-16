"""`progress` events are rate-limited by wall clock; start/end never are.

Driven by a fake clock on the module's own `_monotonic` indirection so nothing
here depends on real timing (see AGENTS.md on timing-sensitive tests).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from birdnet.utils import download_progress
from birdnet.utils.download_progress import DownloadProgress, _ProgressThrottle
from birdnet.utils.helper import download_file_tqdm

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


def _fake_clock(step_s: float, start: float = 100.0) -> Iterator[float]:
  now = start
  while True:
    yield now
    now += step_s


def test_throttle_reports_first_call_then_only_after_the_interval(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ticks = iter([100.0, 100.02, 100.05, 100.11, 100.19, 100.30])
  monkeypatch.setattr(download_progress, "_monotonic", lambda: next(ticks))

  throttle = _ProgressThrottle(0.1)

  assert [throttle.should_report() for _ in range(6)] == [
    True,
    False,
    False,
    True,
    False,
    True,
  ]


def test_progress_events_are_throttled_but_started_and_finished_are_not(
  server: LocalServer,
  events: list[DownloadProgress],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # 4 chunks arrive 60 ms apart on the fake clock; at a 100 ms interval only
  # the 1st and 3rd chunk report. `_monotonic` is consulted once per chunk.
  monkeypatch.setattr(download_progress, "_PROGRESS_MIN_INTERVAL_S", 0.1)
  clock = _fake_clock(0.06)
  monkeypatch.setattr(download_progress, "_monotonic", lambda: next(clock))

  download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert [e.status for e in events] == [
    "started",
    "progress",
    "progress",
    "finished",
  ]
  assert [e.bytes_done for e in events if e.status == "progress"] == [1024, 3072]
  assert events[-1].bytes_done == len(server.body)


def test_throttle_restarts_with_every_attempt(
  server: LocalServer,
  events: list[DownloadProgress],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # A huge interval: only the first chunk of each attempt may report, so a
  # retry must not inherit the previous attempt's "already reported" state.
  monkeypatch.setattr(download_progress, "_PROGRESS_MIN_INTERVAL_S", 1e9)

  download_file_tqdm(server.url("/flaky/1"), tmp_path / "f.bin")

  progress = [e for e in events if e.status == "progress"]
  assert [e.attempt for e in progress] == [1, 2]
