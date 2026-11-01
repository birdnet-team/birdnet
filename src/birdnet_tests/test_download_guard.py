"""Outside the ``load_model`` phase every download is refused at the transport."""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.queues
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn
from unittest import mock

import pytest
import requests

from birdnet.globals import ENV_VAR_OFFLINE
from birdnet.utils import downloads
from birdnet.utils.downloads import DownloadError, download_file_tqdm

pytestmark = pytest.mark.no_tf

_URL = "https://zenodo.org/records/0/files/nothing.zip"


def _fails_with(reason: str) -> Callable[..., NoReturn]:
  def fail(*_args: object, **_kwargs: object) -> NoReturn:
    pytest.fail(reason)

  return fail


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


def test_the_harness_disables_downloads_for_an_unmarked_test() -> None:
  assert os.environ.get(ENV_VAR_OFFLINE) == "1"


@pytest.mark.load_model
def test_the_harness_leaves_downloads_on_for_a_load_model_test() -> None:
  assert os.environ.get(ENV_VAR_OFFLINE, "0") in ("", "0")


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
