"""`failed` is terminal: emitted exactly once, right before the error is raised."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from birdnet.utils import helper
from birdnet.utils.download_progress import DownloadProgress
from birdnet.utils.helper import (
  _DOWNLOAD_ATTEMPTS,
  _DOWNLOAD_RETRY_WAITS_S,
  DownloadError,
  download_file_tqdm,
)

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


def _statuses(events: list[DownloadProgress]) -> list[str]:
  return [e.status for e in events if e.status != "progress"]


def test_client_error_fails_terminally_on_the_first_attempt(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  target = tmp_path / "f.bin"

  with pytest.raises(ValueError, match="Status code: 404"):
    download_file_tqdm(
      server.url("/status/404"), target, download_size=len(server.body)
    )

  assert server.hits_for("/status/404") == 1
  assert _statuses(events) == ["started", "failed"]
  failed = events[-1]
  assert failed.is_terminal
  assert failed.attempt == 1
  assert failed.retry_in_s is None
  assert failed.error is not None
  assert "404" in failed.error
  assert not target.exists()
  assert not list(tmp_path.glob("*.tmp"))


def test_exhausted_attempts_fail_terminally_after_the_last_retry(
  server: LocalServer, events: list[DownloadProgress], tmp_path: Path
) -> None:
  with pytest.raises(DownloadError) as excinfo:
    download_file_tqdm(server.url("/status/503"), tmp_path / "f.bin")

  assert excinfo.value.status_code == 503
  assert server.hits_for("/status/503") == _DOWNLOAD_ATTEMPTS

  started = [e for e in events if e.status == "started"]
  retrying = [e for e in events if e.status == "retrying"]
  failed = [e for e in events if e.status == "failed"]
  assert [e.attempt for e in started] == list(range(1, _DOWNLOAD_ATTEMPTS + 1))
  assert [e.attempt for e in retrying] == list(range(1, _DOWNLOAD_ATTEMPTS))
  assert [e.retry_in_s for e in retrying] == list(_DOWNLOAD_RETRY_WAITS_S)
  assert [e.attempt for e in failed] == [_DOWNLOAD_ATTEMPTS]
  assert events[-1].status == "failed"
  assert not any(e.status == "finished" for e in events)
  assert not (tmp_path / "f.bin").exists()
  assert not list(tmp_path.glob("*.tmp"))


def test_connection_error_before_any_response_is_reported_and_raised(
  events: list[DownloadProgress], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def refuse(*_args: object, **_kwargs: object) -> None:
    raise requests.ConnectionError("always down")

  monkeypatch.setattr(requests, "get", refuse)

  with pytest.raises(requests.ConnectionError):
    download_file_tqdm("http://example.invalid/f", tmp_path / "f.bin")

  assert _statuses(events) == [
    *(["started", "retrying"] * (_DOWNLOAD_ATTEMPTS - 1)),
    "started",
    "failed",
  ]
  assert all(e.bytes_done == 0 for e in events)
  assert events[-1].error is not None
  assert "always down" in events[-1].error


def test_response_is_closed_even_if_content_length_header_is_malformed(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # A response acquired but never closed leaks the connection on every failed
  # attempt; header parsing happens inside the block that closes it.
  closes: list[bool] = []

  class BadHeaderResponse:
    status_code = 200
    headers = {"content-length": "not-a-number"}

    def iter_content(self, _block_size: int) -> list[bytes]:
      return [b"x"]

    def close(self) -> None:
      closes.append(True)

  monkeypatch.setattr(requests, "get", lambda *_a, **_k: BadHeaderResponse())

  with pytest.raises(ValueError):
    download_file_tqdm("http://example.invalid/f", tmp_path / "f.bin")

  assert len(closes) == helper._DOWNLOAD_ATTEMPTS
