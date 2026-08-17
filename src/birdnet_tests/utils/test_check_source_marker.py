from pathlib import Path

from birdnet.utils.helper import (
  SOURCE_MARKER_NAME,
  check_source_marker,
  write_source_marker,
)

_URL = "https://example.org/records/1/files/model.zip"


def test_missing_marker_is_not_available(tmp_path: Path) -> None:
  assert not check_source_marker(tmp_path, _URL)


def test_matching_marker_is_available(tmp_path: Path) -> None:
  write_source_marker(tmp_path, _URL)

  assert check_source_marker(tmp_path, _URL)


def test_marker_from_another_release_is_not_available(tmp_path: Path) -> None:
  write_source_marker(tmp_path, "https://example.org/records/1/files/old.zip")

  assert not check_source_marker(tmp_path, _URL)


def test_trailing_whitespace_is_ignored(tmp_path: Path) -> None:
  (tmp_path / SOURCE_MARKER_NAME).write_text(f"{_URL}\n", encoding="utf-8")

  assert check_source_marker(tmp_path, _URL)


def test_marker_directory_is_not_a_marker(tmp_path: Path) -> None:
  (tmp_path / SOURCE_MARKER_NAME).mkdir()

  assert not check_source_marker(tmp_path, _URL)
