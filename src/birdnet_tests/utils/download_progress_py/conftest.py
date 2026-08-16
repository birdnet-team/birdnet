"""Fixtures for the download-progress contract tests.

The downloads run against a stdlib HTTP server on 127.0.0.1 so the real
``download_file_tqdm`` -> ``_download_file_once`` -> ``requests`` path is
exercised, including the retry loop; only the back-off sleep and the report
throttle are neutralised.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from birdnet.utils import download_progress, helper
from birdnet.utils.download_progress import DownloadProgress

BODY = b"x" * 4096  # 4 chunks of the helper's 1 KiB block size
ERROR_BODY = b"error page"


@dataclass
class LocalServer:
  base_url: str
  body: bytes = BODY
  error_body: bytes = ERROR_BODY
  hits: dict[str, int] = field(default_factory=dict)

  def url(self, path: str) -> str:
    return self.base_url + path

  def hits_for(self, path: str) -> int:
    return self.hits.get(path, 0)


@pytest.fixture
def server() -> Generator[LocalServer, None, None]:
  """Routes (all HTTP/1.0, so the connection closes after each response):

  ``/file``            200, Content-Length, BODY
  ``/no-length``       200, no Content-Length (close-delimited), BODY
  ``/empty``           200, Content-Length 0
  ``/status/<code>``   that status with a small body
  ``/flaky/<n>``       first n hits 503, afterwards like /file
  ``/truncated``       Content-Length twice the body, then close mid-stream
  ``/truncated-once``  first hit like /truncated, afterwards like /file
  """
  local = LocalServer(base_url="")

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_: str, *args: object) -> None:
      pass

    def _send(self, status: int, body: bytes, *, content_length: int | None) -> None:
      self.send_response(status)
      if content_length is not None:
        self.send_header("Content-Length", str(content_length))
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self) -> None:
      path = self.path
      local.hits[path] = local.hits.get(path, 0) + 1
      hit = local.hits[path]
      if path == "/file":
        self._send(200, BODY, content_length=len(BODY))
      elif path == "/no-length":
        self._send(200, BODY, content_length=None)
      elif path == "/empty":
        self._send(200, b"", content_length=0)
      elif path.startswith("/status/"):
        self._send(
          int(path.rsplit("/", 1)[1]), ERROR_BODY, content_length=len(ERROR_BODY)
        )
      elif path.startswith("/flaky/"):
        if hit <= int(path.rsplit("/", 1)[1]):
          self._send(503, ERROR_BODY, content_length=len(ERROR_BODY))
        else:
          self._send(200, BODY, content_length=len(BODY))
      elif path == "/truncated" or (path == "/truncated-once" and hit == 1):
        self._send(200, BODY, content_length=2 * len(BODY))
      elif path == "/truncated-once":
        self._send(200, BODY, content_length=len(BODY))
      else:
        self._send(404, ERROR_BODY, content_length=len(ERROR_BODY))

  httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  local.base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
  thread = threading.Thread(target=httpd.serve_forever, daemon=True)
  thread.start()
  try:
    yield local
  finally:
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _isolated_callback() -> Generator[None, None, None]:
  # The registry is process-wide and xdist workers reuse their process.
  download_progress.set_download_progress_callback(None)
  yield
  download_progress.set_download_progress_callback(None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(helper.time, "sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
  # Every chunk reports, so byte accounting is deterministic. The throttle
  # itself is tested with a fake clock in test_throttle.py.
  monkeypatch.setattr(download_progress, "_PROGRESS_MIN_INTERVAL_S", 0.0)


@pytest.fixture
def events() -> list[DownloadProgress]:
  recorded: list[DownloadProgress] = []
  download_progress.set_download_progress_callback(recorded.append)
  return recorded
