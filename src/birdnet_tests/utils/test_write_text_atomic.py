from pathlib import Path

import pytest

from birdnet.utils.helper import write_text_atomic


@pytest.mark.no_tf
def test_writes_the_content(tmp_path: Path) -> None:
  target = tmp_path / "out.txt"

  write_text_atomic(target, "hello world")

  assert target.read_text(encoding="utf-8") == "hello world"


@pytest.mark.no_tf
def test_leaves_no_temp_files(tmp_path: Path) -> None:
  write_text_atomic(tmp_path / "out.txt", "hello world")

  assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.no_tf
def test_keeps_the_previous_content_when_writing_fails(tmp_path: Path) -> None:
  """A reader sees either the old content or the new one, never a partial file."""
  target = tmp_path / "out.txt"
  target.write_text("original", encoding="utf-8")

  class _Unwritable:
    def __str__(self) -> str:
      raise RuntimeError("boom")

  with pytest.raises(TypeError):
    write_text_atomic(target, _Unwritable())  # type: ignore[arg-type]

  assert target.read_text(encoding="utf-8") == "original"
  assert list(tmp_path.glob("*.tmp")) == []
