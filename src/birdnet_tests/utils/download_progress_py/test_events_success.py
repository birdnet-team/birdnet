"""A clean download: started -> progress* -> finished, with exact byte accounting."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.utils.download_progress import (
  DownloadProgress,
  set_download_progress_callback,
)
from birdnet.utils.helper import _DOWNLOAD_ATTEMPTS, download_file_tqdm

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


def test_started_progress_finished_with_accurate_byte_accounting(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  target = tmp_path / "model.bin"

  result = download_file_tqdm(
    server.url("/file"),
    target,
    download_size=len(server.body),
    description="test model",
  )

  assert result == len(server.body)
  assert target.read_bytes() == server.body

  assert events[0].status == "started"
  assert events[0].bytes_done == 0
  assert events[0].bytes_total == len(server.body)
  assert events[0].attempt == 1
  assert events[0].max_attempts == _DOWNLOAD_ATTEMPTS
  assert events[0].description == "test model"
  assert events[0].url == server.url("/file")
  assert events[0].error is None
  assert events[0].retry_in_s is None

  progress = [e for e in events if e.status == "progress"]
  assert len(progress) == len(server.body) // 1024  # one per 1 KiB chunk (no throttle)
  byte_counts = [e.bytes_done for e in progress]
  assert byte_counts == sorted(byte_counts)
  assert byte_counts[-1] == len(server.body)
  assert all(e.bytes_total == len(server.body) for e in progress)
  assert all(e.attempt == 1 for e in progress)

  assert events[-1].status == "finished"
  assert events[-1].bytes_done == len(server.body)
  assert events[-1].bytes_total == len(server.body)
  assert events[-1].attempt == 1

  assert [e.is_terminal for e in events].count(True) == 1
  assert [e.status for e in events] == ["started", *(["progress"] * 4), "finished"]


def test_description_falls_back_to_the_url(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert all(e.description == server.url("/file") for e in events)


def test_no_events_without_a_registered_callback(
  server: LocalServer, tmp_path: Path
) -> None:
  # `events` fixture not requested: nothing registered. Just has to work.
  result = download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert result == len(server.body)
  assert (tmp_path / "f.bin").read_bytes() == server.body


def test_callback_is_captured_when_the_download_starts(
  server: LocalServer, tmp_path: Path
) -> None:
  # Registering a callback while a download runs must not switch targets
  # half-way; it takes effect from the next download on.
  first: list[DownloadProgress] = []
  second: list[DownloadProgress] = []

  def switching(p: DownloadProgress) -> None:
    first.append(p)
    set_download_progress_callback(second.append)

  set_download_progress_callback(switching)
  download_file_tqdm(server.url("/file"), tmp_path / "a.bin")
  download_file_tqdm(server.url("/file"), tmp_path / "b.bin")

  assert [e.status for e in first][0] == "started"
  assert [e.status for e in first][-1] == "finished"
  assert all(e.url.endswith("/file") for e in first)
  assert second[0].status == "started"
  assert second[-1].status == "finished"
  assert len(first) == len(second)
