"""A raising callback aborts the download cleanly and is never retried.

This is also the documented way to cancel a download from a UI: the caller's
own exception object propagates out of `download_file_tqdm` (and thus out of
`load(..)`), the partial file is discarded, and no further events are sent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.utils import helper
from birdnet.utils.download_progress import (
  DownloadProgress,
  DownloadStatus,
  set_download_progress_callback,
)
from birdnet.utils.helper import download_file_tqdm

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


class Cancelled(Exception):
  pass


class HardStop(BaseException):
  pass


def _raise_on(
  status: DownloadStatus, exc: BaseException, seen: list[DownloadProgress]
) -> None:
  def callback(p: DownloadProgress) -> None:
    seen.append(p)
    if p.status == status:
      raise exc

  set_download_progress_callback(callback)


@pytest.mark.parametrize("status", ["started", "progress"])
def test_exception_propagates_unchanged_and_the_download_is_not_retried(
  status: DownloadStatus, server: LocalServer, tmp_path: Path
) -> None:
  seen: list[DownloadProgress] = []
  error = Cancelled("user pressed cancel")
  _raise_on(status, error, seen)
  target = tmp_path / "f.bin"

  with pytest.raises(Cancelled) as excinfo:
    download_file_tqdm(server.url("/file"), target, download_size=len(server.body))

  assert excinfo.value is error
  assert server.hits_for("/file") == (0 if status == "started" else 1)
  assert seen[-1].status == status  # no event after the one that raised
  assert not target.exists()
  assert not list(tmp_path.glob("*.tmp"))


def test_value_error_from_the_callback_is_not_mistaken_for_a_download_fault(
  server: LocalServer, tmp_path: Path
) -> None:
  # The retry loop catches ValueError (DownloadError is one); a callback
  # raising ValueError must still abort after a single attempt.
  seen: list[DownloadProgress] = []
  _raise_on("progress", ValueError("ui bug"), seen)

  with pytest.raises(ValueError, match="ui bug"):
    download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert server.hits_for("/file") == 1
  assert not (tmp_path / "f.bin").exists()
  assert not list(tmp_path.glob("*.tmp"))


def test_raise_on_retrying_cancels_before_the_back_off_sleep(
  server: LocalServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  slept: list[float] = []
  monkeypatch.setattr(helper.time, "sleep", slept.append)
  seen: list[DownloadProgress] = []
  _raise_on("retrying", Cancelled(), seen)

  with pytest.raises(Cancelled):
    download_file_tqdm(server.url("/status/503"), tmp_path / "f.bin")

  assert server.hits_for("/status/503") == 1
  assert slept == []
  assert seen[-1].status == "retrying"


def test_raise_on_finished_leaves_the_complete_file_in_place(
  server: LocalServer, tmp_path: Path
) -> None:
  seen: list[DownloadProgress] = []
  _raise_on("finished", Cancelled(), seen)

  with pytest.raises(Cancelled):
    download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert (tmp_path / "f.bin").read_bytes() == server.body
  assert not list(tmp_path.glob("*.tmp"))


def test_base_exception_from_the_callback_removes_the_partial_file(
  server: LocalServer, tmp_path: Path
) -> None:
  seen: list[DownloadProgress] = []
  _raise_on("progress", HardStop(), seen)

  with pytest.raises(HardStop):
    download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert not (tmp_path / "f.bin").exists()
  assert not list(tmp_path.glob("*.tmp"))


def test_no_failed_event_is_sent_to_a_callback_that_raised(
  server: LocalServer, tmp_path: Path
) -> None:
  seen: list[DownloadProgress] = []
  _raise_on("progress", Cancelled(), seen)

  with pytest.raises(Cancelled):
    download_file_tqdm(server.url("/file"), tmp_path / "f.bin")

  assert [e.status for e in seen].count("failed") == 0
  assert [e.status for e in seen].count("retrying") == 0
