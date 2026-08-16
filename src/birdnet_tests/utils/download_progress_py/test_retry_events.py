"""Transient faults: `retrying` before each back-off, a fresh `started` per attempt.

The event stream must let a UI tell "failed, retrying in N s" from "failed for
good" at the moment the event arrives, and a restart must be announced rather
than showing up as a silently shrinking byte count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.utils.download_progress import DownloadProgress
from birdnet.utils.helper import _DOWNLOAD_RETRY_WAITS_S, download_file_tqdm

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


def _statuses(events: list[DownloadProgress]) -> list[str]:
  return [e.status for e in events if e.status != "progress"]


def test_two_server_errors_then_success(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  result = download_file_tqdm(
    server.url("/flaky/2"),
    tmp_path / "f.bin",
    download_size=len(server.body),
    description="flaky model",
  )

  assert result == len(server.body)
  assert (tmp_path / "f.bin").read_bytes() == server.body
  assert server.hits_for("/flaky/2") == 3

  assert _statuses(events) == [
    "started",
    "retrying",
    "started",
    "retrying",
    "started",
    "finished",
  ]

  retrying = [e for e in events if e.status == "retrying"]
  assert [e.attempt for e in retrying] == [1, 2]
  assert [e.retry_in_s for e in retrying] == list(_DOWNLOAD_RETRY_WAITS_S[:2])
  assert all(e.error is not None and "503" in e.error for e in retrying)
  assert all(not e.is_terminal for e in retrying)
  # The bytes of the failed attempt (here: the error page) are reported as-is.
  assert all(e.bytes_done == len(server.error_body) for e in retrying)

  started = [e for e in events if e.status == "started"]
  assert [e.attempt for e in started] == [1, 2, 3]
  assert all(e.bytes_done == 0 for e in started)

  assert events[-1].status == "finished"
  assert events[-1].attempt == 3
  assert events[-1].bytes_done == len(server.body)


def test_bytes_never_decrease_within_an_attempt(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  download_file_tqdm(server.url("/flaky/1"), tmp_path / "f.bin")

  by_attempt: dict[int, list[int]] = {}
  for e in events:
    by_attempt.setdefault(e.attempt, []).append(e.bytes_done)

  assert sorted(by_attempt) == [1, 2]
  for counts in by_attempt.values():
    assert counts[0] == 0  # the attempt's "started"
    assert counts == sorted(counts)


def test_truncated_stream_is_reported_with_partial_bytes_then_retried(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  # The connection drops mid-body: a RequestException raised from inside
  # `iter_content`, i.e. after "progress" events already went out.
  download_file_tqdm(
    server.url("/truncated-once"), tmp_path / "f.bin", download_size=len(server.body)
  )

  assert _statuses(events) == ["started", "retrying", "started", "finished"]
  retrying = next(e for e in events if e.status == "retrying")
  assert retrying.attempt == 1
  assert retrying.bytes_done == len(server.body)  # everything that did arrive
  assert retrying.error is not None
  assert "IncompleteRead" in retrying.error or "Connection broken" in retrying.error
  assert (tmp_path / "f.bin").read_bytes() == server.body
  assert not list(tmp_path.glob("*.tmp"))
