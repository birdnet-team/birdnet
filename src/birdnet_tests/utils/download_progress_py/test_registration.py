"""Registration API: setter/getter/context manager and the payload's helpers."""

from __future__ import annotations

import pytest

from birdnet.utils import download_progress
from birdnet.utils.download_progress import (
  DownloadProgress,
  download_progress_callback,
  get_download_progress_callback,
  set_download_progress_callback,
)

pytestmark = [pytest.mark.no_tf]


def _cb_a(_p: DownloadProgress) -> None:
  pass


def _cb_b(_p: DownloadProgress) -> None:
  pass


def test_setter_returns_previous_and_none_clears() -> None:
  assert get_download_progress_callback() is None
  assert set_download_progress_callback(_cb_a) is None
  assert get_download_progress_callback() is _cb_a
  assert set_download_progress_callback(_cb_b) is _cb_a
  assert set_download_progress_callback(None) is _cb_b
  assert get_download_progress_callback() is None


def test_context_manager_restores_previous_on_exit_and_on_exception() -> None:
  set_download_progress_callback(_cb_a)

  with download_progress_callback(_cb_b):
    assert get_download_progress_callback() is _cb_b
  assert get_download_progress_callback() is _cb_a

  with pytest.raises(RuntimeError), download_progress_callback(_cb_b):
    raise RuntimeError("body failed")
  assert get_download_progress_callback() is _cb_a


def test_nested_context_managers_restore_in_order() -> None:
  with download_progress_callback(_cb_a):
    with download_progress_callback(_cb_b):
      assert get_download_progress_callback() is _cb_b
    assert get_download_progress_callback() is _cb_a
  assert get_download_progress_callback() is None


def test_public_names_are_exported_from_the_package() -> None:
  import birdnet

  assert birdnet.DownloadProgress is DownloadProgress
  assert birdnet.set_download_progress_callback is set_download_progress_callback
  assert birdnet.get_download_progress_callback is get_download_progress_callback
  assert birdnet.download_progress_callback is download_progress_callback
  assert birdnet.DownloadStatus is download_progress.DownloadStatus
  assert birdnet.DownloadProgressCallback is download_progress.DownloadProgressCallback


def _progress(**overrides: object) -> DownloadProgress:
  values: dict[str, object] = {
    "description": "d",
    "url": "u",
    "bytes_done": 0,
    "bytes_total": None,
    "attempt": 1,
    "max_attempts": 5,
    "status": "started",
  }
  values.update(overrides)
  return DownloadProgress(**values)  # type: ignore[arg-type]


def test_fraction_is_none_while_total_unknown_and_clamped_otherwise() -> None:
  assert _progress(bytes_done=10, bytes_total=None).fraction is None
  assert _progress(bytes_done=0, bytes_total=0).fraction is None
  assert _progress(bytes_done=25, bytes_total=100).fraction == 0.25
  assert _progress(bytes_done=150, bytes_total=100).fraction == 1.0


@pytest.mark.parametrize(
  ("status", "terminal"),
  [
    ("started", False),
    ("progress", False),
    ("retrying", False),
    ("finished", True),
    ("failed", True),
  ],
)
def test_is_terminal(status: str, terminal: bool) -> None:
  assert _progress(status=status).is_terminal is terminal


def test_payload_is_frozen() -> None:
  with pytest.raises(AttributeError):
    _progress().bytes_done = 1  # type: ignore[misc]
