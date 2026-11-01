"""Outside the ``load_model`` phase every download is refused at the transport."""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.queues
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast
from unittest import mock

import pytest
import requests

from birdnet.globals import ENV_VAR_OFFLINE
from birdnet.utils import downloads
from birdnet.utils.download_progress import (
  DownloadProgress,
  download_progress_callback,
)
from birdnet.utils.downloads import DownloadError, download_file_tqdm
from birdnet_tests import conftest

pytestmark = pytest.mark.no_tf

_URL = "https://zenodo.org/records/0/files/nothing.zip"


def _fails_with(reason: str) -> Callable[..., NoReturn]:
  def fail(*_args: object, **_kwargs: object) -> NoReturn:
    pytest.fail(reason)

  return fail


def _set_offline(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
  if value is None:
    monkeypatch.delenv(ENV_VAR_OFFLINE, raising=False)
  else:
    monkeypatch.setenv(ENV_VAR_OFFLINE, value)


class _GotPastTheCheckError(Exception):
  pass


def test_the_library_refuses_before_any_request_and_never_retries(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_OFFLINE, "1")
  monkeypatch.setattr(requests, "get", _fails_with("requests.get was called"))
  monkeypatch.setattr(downloads.time, "sleep", _fails_with("the refusal was retried"))
  destination = tmp_path / "nothing.zip"

  with pytest.raises(DownloadError, match=re.escape(_URL)) as raised:
    download_file_tqdm(_URL, destination)

  assert ENV_VAR_OFFLINE in str(raised.value)
  assert str(destination) in str(raised.value)
  assert not destination.exists()
  assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
  ("value", "refuses"),
  [("1", True), ("true", True), ("0", False), ("", False), (None, False)],
  ids=["1", "true", "0", "empty", "absent"],
)
def test_any_value_but_0_or_empty_refuses(
  value: str | None, refuses: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _set_offline(monkeypatch, value)
  destination = tmp_path / "nothing.zip"

  if refuses:
    monkeypatch.setattr(requests, "get", _fails_with("requests.get was called"))
    with pytest.raises(DownloadError, match=ENV_VAR_OFFLINE):
      downloads._download_file_once(_URL, destination)
  else:
    monkeypatch.setattr(requests, "get", mock.Mock(side_effect=_GotPastTheCheckError))
    with pytest.raises(_GotPastTheCheckError):
      downloads._download_file_once(_URL, destination)


def test_a_refused_download_reports_only_its_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv(ENV_VAR_OFFLINE, "1")
  monkeypatch.setattr(requests, "get", _fails_with("requests.get was called"))
  monkeypatch.setattr(downloads.time, "sleep", _fails_with("the refusal was retried"))
  events: list[DownloadProgress] = []

  with download_progress_callback(events.append), pytest.raises(DownloadError):
    download_file_tqdm(_URL, tmp_path / "nothing.zip")

  assert [event.status for event in events] == ["failed"]
  (failed,) = events
  assert failed.attempt == 0
  assert failed.bytes_done == 0
  assert failed.bytes_total is None
  assert failed.error is not None
  assert ENV_VAR_OFFLINE in failed.error


def test_the_harness_disables_downloads_for_an_unmarked_test() -> None:
  assert os.environ.get(ENV_VAR_OFFLINE) == "1"


@pytest.fixture(scope="module")
def _offline_when_module_fixtures_are_set_up() -> str | None:
  return os.environ.get(ENV_VAR_OFFLINE)


def test_the_harness_disables_downloads_before_module_fixtures(
  _offline_when_module_fixtures_are_set_up: str | None,
) -> None:
  assert _offline_when_module_fixtures_are_set_up == "1"


@pytest.mark.load_model
def test_the_harness_leaves_downloads_on_for_a_load_model_test() -> None:
  assert os.environ.get(ENV_VAR_OFFLINE, "0") in ("", "0")


class _StubItem:
  """What the harness hooks read from a test item."""

  def __init__(self, marker: pytest.Mark | None) -> None:
    self.stash = pytest.Stash()
    self._marker = marker

  def get_closest_marker(self, name: str) -> pytest.Mark | None:
    return self._marker


@pytest.mark.parametrize("before", [None, "0"], ids=["absent", "0"])
def test_the_harness_restores_the_variable_after_an_unmarked_test(
  before: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
  _set_offline(monkeypatch, before)
  item = cast(pytest.Item, _StubItem(marker=None))

  conftest.pytest_runtest_setup(item)
  assert os.environ.get(ENV_VAR_OFFLINE) == "1"
  conftest.pytest_runtest_teardown(item)
  assert os.environ.get(ENV_VAR_OFFLINE) == before


@pytest.mark.parametrize("before", [None, "0"], ids=["absent", "0"])
def test_the_harness_leaves_the_variable_alone_for_a_load_model_item(
  before: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
  _set_offline(monkeypatch, before)
  item = cast(pytest.Item, _StubItem(marker=pytest.mark.load_model.mark))

  conftest.pytest_runtest_setup(item)
  assert os.environ.get(ENV_VAR_OFFLINE) == before
  conftest.pytest_runtest_teardown(item)
  assert os.environ.get(ENV_VAR_OFFLINE) == before


def _download_in_child(
  url: str, destination: str, results: multiprocessing.queues.Queue[tuple[str, str]]
) -> None:
  with mock.patch.object(
    requests, "get", side_effect=AssertionError("requests.get was called")
  ):
    try:
      download_file_tqdm(url, Path(destination))
    except DownloadError as error:
      results.put(("DownloadError", str(error)))
    except BaseException as error:
      results.put((type(error).__name__, str(error)))
    else:
      results.put(("no error", ""))


def test_a_spawned_child_inherits_the_refusal(tmp_path: Path) -> None:
  context = mp.get_context("spawn")
  results: multiprocessing.queues.Queue[tuple[str, str]] = context.Queue()
  child = context.Process(
    target=_download_in_child, args=(_URL, str(tmp_path / "nothing.zip"), results)
  )
  child.start()
  try:
    name, message = results.get(timeout=120)
  finally:
    child.join(timeout=30)
    if child.is_alive():
      child.kill()
      child.join()

  assert name == "DownloadError", message
  assert ENV_VAR_OFFLINE in message
