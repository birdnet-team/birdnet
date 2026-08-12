"""`download_file_tqdm` must retry transient faults and fail fast on 4xx.

Every official model/label/taxonomy download goes through this helper, so a
single connection reset or truncated stream used to fail the whole `load()`.
The retry loop is unit-tested here with the actual download stubbed out; the
real-download path is covered by `test_download_file_tqdm.py`.
"""

from pathlib import Path

import pytest
import requests

from birdnet.utils import helper


def _http_error(status_code: int) -> requests.HTTPError:
  response = requests.Response()
  response.status_code = status_code
  return requests.HTTPError(f"status {status_code}", response=response)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(helper.time, "sleep", lambda _s: None)


def test_transient_connection_errors_are_retried_until_success(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  calls: list[int] = []

  def fake_download(*args: object, **kwargs: object) -> int:
    calls.append(1)
    if len(calls) < 3:
      raise requests.ConnectionError("reset by peer")
    return 123

  monkeypatch.setattr(helper, "_download_file_once", fake_download)

  result = helper.download_file_tqdm("https://example.org/f", tmp_path / "f")

  assert result == 123
  assert len(calls) == 3


def test_truncated_download_is_retried(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  calls: list[int] = []

  def fake_download(*args: object, **kwargs: object) -> int:
    calls.append(1)
    if len(calls) < 2:
      raise ValueError("Expected size: 10 bytes, downloaded size: 3 bytes.")
    return 10

  monkeypatch.setattr(helper, "_download_file_once", fake_download)

  assert helper.download_file_tqdm("https://example.org/f", tmp_path / "f") == 10
  assert len(calls) == 2


def test_client_error_is_not_retried(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  calls: list[int] = []

  def fake_download(*args: object, **kwargs: object) -> int:
    calls.append(1)
    raise _http_error(404)

  monkeypatch.setattr(helper, "_download_file_once", fake_download)

  with pytest.raises(requests.HTTPError):
    helper.download_file_tqdm("https://example.org/f", tmp_path / "f")

  assert len(calls) == 1


def test_server_error_is_retried_and_raised_when_attempts_are_exhausted(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  calls: list[int] = []

  def fake_download(*args: object, **kwargs: object) -> int:
    calls.append(1)
    raise _http_error(503)

  monkeypatch.setattr(helper, "_download_file_once", fake_download)

  with pytest.raises(requests.HTTPError):
    helper.download_file_tqdm("https://example.org/f", tmp_path / "f")

  assert len(calls) == helper._DOWNLOAD_ATTEMPTS
