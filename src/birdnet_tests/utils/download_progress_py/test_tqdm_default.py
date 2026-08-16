"""The tqdm bar is untouched without a callback and silenced with one."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.utils import helper
from birdnet.utils.download_progress import set_download_progress_callback
from birdnet.utils.helper import download_file_tqdm

from .conftest import LocalServer

pytestmark = [pytest.mark.no_tf]


def test_tqdm_is_disabled_only_while_a_callback_is_registered(
  server: LocalServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  captured_disable: list[bool] = []

  class SpyTqdm(helper.tqdm):
    def __init__(self, *args: object, **kwargs: object) -> None:
      captured_disable.append(bool(kwargs.get("disable")))
      super().__init__(*args, **kwargs)  # type: ignore[arg-type]

  monkeypatch.setattr(helper, "tqdm", SpyTqdm)

  download_file_tqdm(server.url("/file"), tmp_path / "no_cb.bin")
  set_download_progress_callback(lambda _p: None)
  download_file_tqdm(server.url("/file"), tmp_path / "with_cb.bin")
  set_download_progress_callback(None)
  download_file_tqdm(server.url("/file"), tmp_path / "no_cb_again.bin")

  assert captured_disable == [False, True, False]


def test_default_path_still_renders_the_bar_on_stderr(
  server: LocalServer, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
  download_file_tqdm(
    server.url("/file"), tmp_path / "f.bin", description="Downloading test model"
  )

  assert "Downloading test model" in capfd.readouterr().err
