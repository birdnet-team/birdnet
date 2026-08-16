"""Contract tests for `birdnet.set_download_progress_callback`.

A GUI embedding birdnet (e.g. BirdNET-Analyzer) has no visibility into the
tqdm bar `download_file_tqdm` writes to stderr, so it needs a callback to
drive its own progress UI. These tests pin the contract: a "started" event
before any byte is written, throttled "progress" events with an accurate
byte count, exactly one terminal "finished"/"failed" event per attempt, a
fresh "started" (not a silent byte-count regression) on retry, and that a
misbehaving callback can neither block nor corrupt the download.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from birdnet.utils import helper
from birdnet.utils.helper import DownloadProgress, download_file_tqdm


@pytest.fixture(autouse=True)
def _reset_download_progress_callback() -> Generator[None, None, None]:
  helper.set_download_progress_callback(None)
  yield
  helper.set_download_progress_callback(None)


@contextmanager
def _running_server(
  body: bytes, *, send_content_length: bool, chunk_size: int = 256
) -> Generator[str, None, None]:
  class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
      self.send_response(200)
      if send_content_length:
        self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      for i in range(0, len(body), chunk_size):
        self.wfile.write(body[i : i + chunk_size])

    def log_message(self, format_: str, *args: object) -> None:
      pass

  server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
  port = server.server_address[1]
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    yield f"http://127.0.0.1:{port}/f"
  finally:
    server.shutdown()
    thread.join(timeout=5)


class _FakeResponse:
  def __init__(
    self,
    chunks: list[bytes],
    *,
    status_code: int = 200,
    content_length: int | None = None,
  ) -> None:
    self.status_code = status_code
    self._chunks = chunks
    self.headers: dict[str, str] = (
      {"content-length": str(content_length)} if content_length is not None else {}
    )

  def iter_content(self, block_size: int) -> Iterator[bytes]:
    yield from self._chunks

  def close(self) -> None:
    pass


def test_reports_started_progress_and_finished_with_accurate_byte_accounting(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # Disable throttling so every chunk produces a "progress" event, making the
  # byte accounting fully deterministic instead of timing-dependent.
  monkeypatch.setattr(helper, "_DOWNLOAD_PROGRESS_MIN_INTERVAL_S", 0.0)
  body = b"x" * 5000
  events: list[DownloadProgress] = []
  helper.set_download_progress_callback(events.append)

  with _running_server(body, send_content_length=True, chunk_size=500) as url:
    result = download_file_tqdm(url, tmp_path / "f.bin", description="test model")

  assert result == len(body)
  assert (tmp_path / "f.bin").read_bytes() == body

  assert events[0].status == "started"
  assert events[0].bytes_done == 0
  assert events[0].attempt == 1
  assert events[0].description == "test model"

  assert events[-1].status == "finished"
  assert events[-1].bytes_done == len(body)
  assert events[-1].bytes_total == len(body)

  progress_events = [e for e in events if e.status == "progress"]
  assert len(progress_events) >= 3
  byte_counts = [e.bytes_done for e in progress_events]
  assert byte_counts == sorted(byte_counts)
  assert byte_counts[-1] <= len(body)
  assert all(e.bytes_total == len(body) for e in progress_events)


def test_unknown_total_size_reports_none(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(helper, "_DOWNLOAD_PROGRESS_MIN_INTERVAL_S", 0.0)
  body = b"y" * 2000
  events: list[DownloadProgress] = []
  helper.set_download_progress_callback(events.append)

  with _running_server(body, send_content_length=False, chunk_size=400) as url:
    download_file_tqdm(url, tmp_path / "f.bin")

  assert len(events) >= 2
  assert all(e.bytes_total is None for e in events)
  assert events[-1].status == "finished"
  assert events[-1].bytes_done == len(body)


def test_download_progress_throttle_only_reports_after_min_interval(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Unit-tests the throttle's own decision against a fake clock, so this
  # cannot flake on a loaded CI runner the way asserting an event count from
  # a real, timed transfer would (see AGENTS.md on timing-sensitive tests).
  fake_times = iter([100.0, 100.02, 100.05, 100.11, 100.19, 100.30])
  monkeypatch.setattr(helper.time, "monotonic", lambda: next(fake_times))

  throttle = helper._DownloadProgressThrottle(0.1)
  results = [throttle.should_report() for _ in range(6)]

  assert results == [True, False, False, True, False, True]


def test_retry_reports_a_fresh_started_event_and_does_not_go_backwards_silently(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(helper.time, "sleep", lambda _s: None)
  events: list[DownloadProgress] = []
  helper.set_download_progress_callback(events.append)

  calls = {"n": 0}

  def fake_get(_url: str, **_kwargs: object) -> _FakeResponse:
    calls["n"] += 1
    if calls["n"] == 1:
      raise requests.ConnectionError("reset by peer")
    return _FakeResponse([b"abcd"], content_length=4)

  monkeypatch.setattr(requests, "get", fake_get)

  result = download_file_tqdm(
    "http://example.invalid/f", tmp_path / "f.bin", description="retry model"
  )

  assert result == 4
  assert calls["n"] == 2

  started_events = [e for e in events if e.status == "started"]
  failed_events = [e for e in events if e.status == "failed"]
  assert [e.attempt for e in started_events] == [1, 2]
  # The second "started" IS the restart notification: attempt goes up and
  # bytes_done resets to 0, so a consumer never sees the count drop silently.
  assert started_events[1].bytes_done == 0
  assert len(failed_events) == 1
  assert failed_events[0].attempt == 1
  assert failed_events[0].error is not None
  assert "reset by peer" in failed_events[0].error
  assert events[-1].status == "finished"
  assert events[-1].attempt == 2


def test_retry_exhausts_all_attempts_then_raises(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(helper.time, "sleep", lambda _s: None)
  events: list[DownloadProgress] = []
  helper.set_download_progress_callback(events.append)

  def fake_get(_url: str, **_kwargs: object) -> _FakeResponse:
    raise requests.ConnectionError("always down")

  monkeypatch.setattr(requests, "get", fake_get)

  with pytest.raises(requests.ConnectionError):
    download_file_tqdm("http://example.invalid/f", tmp_path / "f.bin")

  expected_attempts = list(range(1, helper._DOWNLOAD_ATTEMPTS + 1))
  started_events = [e for e in events if e.status == "started"]
  failed_events = [e for e in events if e.status == "failed"]
  assert [e.attempt for e in started_events] == expected_attempts
  assert [e.attempt for e in failed_events] == expected_attempts
  assert not any(e.status == "finished" for e in events)
  assert not (tmp_path / "f.bin").exists()


def test_final_failure_reports_failed_and_still_raises(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  events: list[DownloadProgress] = []
  helper.set_download_progress_callback(events.append)

  def fake_get(_url: str, **_kwargs: object) -> _FakeResponse:
    return _FakeResponse([b"ab"], status_code=404, content_length=10)

  monkeypatch.setattr(requests, "get", fake_get)

  with pytest.raises(ValueError, match="Status code: 404"):
    download_file_tqdm("http://example.invalid/f", tmp_path / "f.bin")

  # The single 2-byte chunk always produces a "progress" tick -- the throttle
  # never withholds the very first update of an attempt -- before the size
  # mismatch is detected and reported as "failed".
  assert [e.status for e in events] == ["started", "progress", "failed"]
  assert events[-1].error is not None
  assert "404" in events[-1].error
  assert not (tmp_path / "f.bin").exists()


def test_raising_callback_does_not_corrupt_the_download(tmp_path: Path) -> None:
  def bad_callback(_progress: DownloadProgress) -> None:
    raise RuntimeError("boom")

  helper.set_download_progress_callback(bad_callback)

  body = b"w" * 3000
  with _running_server(body, send_content_length=True, chunk_size=300) as url:
    result = download_file_tqdm(url, tmp_path / "f.bin")

  assert result == len(body)
  assert (tmp_path / "f.bin").read_bytes() == body


def test_context_manager_restores_previous_callback() -> None:
  events_a: list[DownloadProgress] = []
  events_b: list[DownloadProgress] = []
  helper.set_download_progress_callback(events_a.append)

  probe = DownloadProgress(
    description="x",
    url="u",
    bytes_done=0,
    bytes_total=None,
    attempt=1,
    max_attempts=1,
    status="started",
  )

  with helper.download_progress_callback(events_b.append):
    helper._report_download_progress(probe)

  helper._report_download_progress(probe)

  assert len(events_b) == 1
  assert len(events_a) == 1
