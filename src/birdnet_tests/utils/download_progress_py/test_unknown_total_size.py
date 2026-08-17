"""`bytes_total` is None when neither `download_size` nor Content-Length is known."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.utils.download_progress import DownloadProgress
from birdnet.utils.helper import download_file_tqdm

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


def test_unknown_total_reports_none_throughout(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  download_file_tqdm(server.url("/no-length"), tmp_path / "f.bin")

  assert len(events) >= 3
  assert all(e.bytes_total is None for e in events)
  assert all(e.fraction is None for e in events)
  assert events[-1].status == "finished"
  assert events[-1].bytes_done == len(server.body)
  assert (tmp_path / "f.bin").read_bytes() == server.body


def test_download_size_supplies_the_total_when_the_header_is_missing(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  download_file_tqdm(
    server.url("/no-length"), tmp_path / "f.bin", download_size=len(server.body)
  )

  assert all(e.bytes_total == len(server.body) for e in events)
  assert events[-1].status == "finished"


def test_content_length_supplies_the_total_after_started(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  # Without `download_size` the total is only known once the headers arrive.
  download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert events[0].status == "started"
  assert events[0].bytes_total is None
  assert all(e.bytes_total == len(server.body) for e in events[1:])


def test_zero_byte_content_length_reports_a_real_zero_not_unknown(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  download_file_tqdm(server.url("/empty"), tmp_path / "f.bin")

  assert events[-1].status == "finished"
  assert events[-1].bytes_total == 0
  assert events[-1].bytes_done == 0
  assert (tmp_path / "f.bin").read_bytes() == b""
