"""A child that failed at the Windows commit limit points at the page file."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import birdnet.acoustic.inference.session as session_mod
from birdnet.acoustic.inference.session import _page_file_hint
from birdnet_tests.fake_acoustic_backend import (
  PageFileBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_PAGE_FILE_TEXT = "The paging file is too small for this operation to complete."
_FAILURE = (
  "OSError: [WinError 1455] The paging file is too small for this operation to "
  'complete. Error loading "torch\\lib\\shm.dll" or one of its dependencies.'
)
_IMPORT_FAILURE = (
  f"ImportError: DLL load failed while importing _multiarray_umath: {_PAGE_FILE_TEXT}"
)
_HINT = (
  " WinError 1455 means the page file is too small for this run: enlarge it or "
  "let Windows manage its size, or lower 'n_workers'."
)


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
  # English text on every platform: the real lookup is Windows-only and
  # localised.
  monkeypatch.setattr(sys, "platform", "win32")
  monkeypatch.setattr(session_mod, "_windows_error_text", lambda code: _PAGE_FILE_TEXT)
  yield


@pytest.mark.usefixtures("windows")
def test_windows_gets_the_hint() -> None:
  assert _page_file_hint(_FAILURE) == _HINT


@pytest.mark.usefixtures("windows")
def test_a_failed_extension_load_gets_the_hint() -> None:
  assert _page_file_hint(_IMPORT_FAILURE) == _HINT


def test_other_platforms_get_none(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "platform", "linux")
  assert _page_file_hint(_FAILURE) == ""
  assert _page_file_hint(_IMPORT_FAILURE) == ""


@pytest.mark.usefixtures("windows")
def test_other_failures_get_none_on_windows() -> None:
  assert _page_file_hint("RuntimeError: bad weights") == ""
  assert _page_file_hint("ImportError: DLL load failed while importing x: y.") == ""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' own error text")
def test_the_systems_own_text_is_matched() -> None:
  # From the system directly, independently of the lookup under test.
  text = ctypes.FormatError(1455).strip()
  assert text
  failure = f"ImportError: DLL load failed while importing _multiarray_umath: {text}"
  assert _page_file_hint(failure) == _HINT


def test_the_caller_gets_it_after_the_childs_failure(tmp_path: Path) -> None:
  wav = write_marked_audio(tmp_path / "in.wav", 4)
  with (
    fake_predict_session(tmp_path, n_workers=1, backend=PageFileBackend) as session,
    pytest.raises(RuntimeError) as info,
  ):
    session.run(str(wav))
  hint = _HINT if sys.platform == "win32" else ""
  assert f"failed: {_FAILURE}{hint}\n\n" in str(info.value), str(info.value)
