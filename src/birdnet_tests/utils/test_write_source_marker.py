from pathlib import Path

from birdnet.utils.helper import SOURCE_MARKER_NAME, write_source_marker

_URL = "https://example.org/records/1/files/model.zip"


def test_writes_the_url_into_the_model_directory(tmp_path: Path) -> None:
  write_source_marker(tmp_path, _URL)

  assert (tmp_path / SOURCE_MARKER_NAME).read_text(encoding="utf-8") == _URL


def test_overwrites_a_marker_from_another_release(tmp_path: Path) -> None:
  write_source_marker(tmp_path, "https://example.org/records/1/files/old.zip")

  write_source_marker(tmp_path, _URL)

  assert (tmp_path / SOURCE_MARKER_NAME).read_text(encoding="utf-8") == _URL
