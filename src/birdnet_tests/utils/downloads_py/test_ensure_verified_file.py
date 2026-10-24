"""ensure_verified_file(..): only bytes that matched their digest reach the path."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from birdnet.utils import downloads, helper
from birdnet.utils.downloads import ensure_verified_file

pytestmark = pytest.mark.no_tf

_URL = "https://example.org/artifact.bin"
_CONTENT = b"the pinned release\n"
_SAME_LENGTH = b"another release!!!\n"
assert len(_SAME_LENGTH) == len(_CONTENT)


@pytest.fixture
def path(tmp_path: Path) -> Path:
  return tmp_path / "cache" / "artifact.bin"


def _serve(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[Path]:
  """Fake the transport: write `content` and record each file written to."""
  written: list[Path] = []

  def fake(
    url: str,
    file_path: Path,
    *,
    download_size: int | None = None,
    description: str | None = None,
  ) -> int:
    written.append(file_path)
    file_path.write_bytes(content)
    return len(content)

  monkeypatch.setattr(helper, "download_file_tqdm", fake)
  return written


def _refuse_rename(monkeypatch: pytest.MonkeyPatch, held: bytes) -> None:
  """Windows: a concurrent process published `held` at the path and holds it."""
  real_replace = os.replace

  def refused(src: str | Path, dst: str | Path) -> None:
    if not str(src).endswith(".unverified"):
      real_replace(src, dst)
      return
    Path(dst).write_bytes(held)
    raise PermissionError("the file is open in another process")

  monkeypatch.setattr(downloads.os, "replace", refused)


def _ensure(path: Path, *, hash_on_hit: bool) -> None:
  ensure_verified_file(
    path,
    _URL,
    len(_CONTENT),
    hashlib.sha256(_CONTENT).hexdigest(),
    "test file",
    hash_on_hit=hash_on_hit,
  )


def test_hash_on_hit_replaces_a_same_length_file_of_other_content(
  path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  path.parent.mkdir()
  path.write_bytes(_SAME_LENGTH)
  written = _serve(monkeypatch, _CONTENT)

  _ensure(path, hash_on_hit=True)

  assert len(written) == 1
  assert path.read_bytes() == _CONTENT


def test_without_hash_on_hit_a_file_of_the_right_size_is_kept(
  path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The caller vouches for the name, so the bytes there are not read."""
  path.parent.mkdir()
  path.write_bytes(_SAME_LENGTH)
  written = _serve(monkeypatch, _CONTENT)

  _ensure(path, hash_on_hit=False)

  assert written == []
  assert path.read_bytes() == _SAME_LENGTH


@pytest.mark.parametrize("hash_on_hit", [True, False])
def test_downloads_without_a_legacy_path(
  path: Path, monkeypatch: pytest.MonkeyPatch, hash_on_hit: bool
) -> None:
  written = _serve(monkeypatch, _CONTENT)

  _ensure(path, hash_on_hit=hash_on_hit)

  assert len(written) == 1
  assert path.read_bytes() == _CONTENT


@pytest.mark.parametrize("hash_on_hit", [True, False])
def test_a_mismatching_download_raises_and_leaves_nothing(
  path: Path, monkeypatch: pytest.MonkeyPatch, hash_on_hit: bool
) -> None:
  _serve(monkeypatch, _SAME_LENGTH)

  with pytest.raises(RuntimeError, match="does not match its expected checksum"):
    _ensure(path, hash_on_hit=hash_on_hit)

  assert not path.exists()
  assert list(path.parent.glob("*.unverified")) == []


def test_downloads_into_a_scratch_file_beside_the_path(
  path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Beside it, so the rename onto `path` never crosses a file system."""
  written = _serve(monkeypatch, _CONTENT)

  _ensure(path, hash_on_hit=True)

  (scratch,) = written
  assert scratch.parent == path.parent
  assert scratch.name.startswith(f"{path.name}.")
  assert scratch.name.endswith(".unverified")
  assert not scratch.exists()


@pytest.mark.parametrize("hash_on_hit", [True, False])
def test_a_refused_rename_onto_the_wanted_bytes_succeeds(
  path: Path, monkeypatch: pytest.MonkeyPatch, hash_on_hit: bool
) -> None:
  written = _serve(monkeypatch, _CONTENT)
  _refuse_rename(monkeypatch, held=_CONTENT)

  _ensure(path, hash_on_hit=hash_on_hit)

  assert len(written) == 1
  assert path.read_bytes() == _CONTENT
  assert list(path.parent.glob("*.unverified")) == []


def test_a_refused_rename_onto_other_content_raises(
  path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """With `hash_on_hit`, a same-length file there is not the wanted one."""
  _serve(monkeypatch, _CONTENT)
  _refuse_rename(monkeypatch, held=_SAME_LENGTH)

  with pytest.raises(PermissionError):
    _ensure(path, hash_on_hit=True)

  assert list(path.parent.glob("*.unverified")) == []
