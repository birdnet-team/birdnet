"""download_zip_members(..): fetch a zip and put its members where they belong."""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from birdnet.utils import helper
from birdnet.utils.downloads import download_zip_members

pytestmark = pytest.mark.no_tf

_URL = "https://example.org/model.zip"


@pytest.fixture
def source_zip(tmp_path: Path) -> Path:
  path = tmp_path / "source.zip"
  with zipfile.ZipFile(path, "w") as archive:
    archive.writestr("model-dir/saved_model.pb", b"model")
    archive.writestr("model-dir/variables/variables.index", b"index")
    archive.writestr("model.tflite", b"tflite")
    archive.writestr("labels/en_us.txt", "Genus_species_Species\n")
  return path


@pytest.fixture
def served(source_zip: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """Fake the transport: serve `source_zip` and record every URL asked for."""
  calls: list[str] = []

  def fake(
    url: str,
    file_path: Path,
    *,
    download_size: int | None = None,
    description: str | None = None,
  ) -> int:
    calls.append(url)
    file_path.write_bytes(source_zip.read_bytes())
    return file_path.stat().st_size

  monkeypatch.setattr(helper, "download_file_tqdm", fake)
  return calls


def _download(members: Sequence[tuple[str | None, Path]]) -> None:
  download_zip_members(_URL, 1234, "a model", members)


def test_a_directory_member_replaces_a_stale_directory(
  tmp_path: Path, served: list[str]
) -> None:
  destination = tmp_path / "models" / "model-fp32"
  destination.mkdir(parents=True)
  (destination / "stale.txt").write_text("from an older release", encoding="utf-8")

  _download([("model-dir", destination)])

  assert served == [_URL]
  assert (destination / "saved_model.pb").read_bytes() == b"model"
  assert (destination / "variables" / "variables.index").read_bytes() == b"index"
  assert not (destination / "stale.txt").exists()


def test_a_file_member_replaces_a_stale_file(tmp_path: Path, served: list[str]) -> None:
  destination = tmp_path / "models" / "model-fp32.tflite"
  destination.parent.mkdir(parents=True)
  destination.write_bytes(b"stale")

  _download([("model.tflite", destination)])

  assert destination.read_bytes() == b"tflite"


def test_the_root_member_moves_the_whole_extraction(
  tmp_path: Path, served: list[str]
) -> None:
  destination = tmp_path / "models" / "model-fp32"

  _download([(None, destination)])

  assert (destination / "model-dir" / "saved_model.pb").read_bytes() == b"model"
  assert (destination / "model.tflite").read_bytes() == b"tflite"
  assert (destination / "labels" / "en_us.txt").is_file()


def test_several_members_land_in_their_places(
  tmp_path: Path, served: list[str]
) -> None:
  model_dir = tmp_path / "models" / "model-fp32"
  lang_dir = tmp_path / "models" / "labels"

  _download([("model-dir", model_dir), ("labels", lang_dir)])

  assert (model_dir / "saved_model.pb").is_file()
  assert (lang_dir / "en_us.txt").is_file()
  assert not (model_dir / "labels").exists()


def test_the_scratch_directory_sits_beside_the_destination(
  tmp_path: Path, source_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A move from there is a rename, so a kill mid-way cannot leave a member
  half copied."""
  downloaded_to: list[Path] = []

  def fake(url: str, file_path: Path, **kwargs: object) -> int:
    downloaded_to.append(file_path)
    file_path.write_bytes(source_zip.read_bytes())
    return file_path.stat().st_size

  monkeypatch.setattr(helper, "download_file_tqdm", fake)
  destination = tmp_path / "models" / "model-fp32"

  _download([(None, destination)])

  assert downloaded_to[0].parent.parent == destination.parent


def test_the_scratch_directory_is_removed(tmp_path: Path, served: list[str]) -> None:
  destination = tmp_path / "models" / "model-fp32"

  _download([(None, destination)])

  assert [path.name for path in destination.parent.iterdir()] == ["model-fp32"]


def test_the_scratch_directory_is_removed_when_extraction_fails(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def serve_garbage(url: str, file_path: Path, **kwargs: object) -> int:
    file_path.write_bytes(b"not a zip")
    return 9

  monkeypatch.setattr(helper, "download_file_tqdm", serve_garbage)
  destination = tmp_path / "models" / "model-fp32"

  with pytest.raises(zipfile.BadZipFile):
    _download([(None, destination)])

  assert list(destination.parent.iterdir()) == []


def test_extraction_is_logged_not_printed(
  tmp_path: Path, served: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
  # Not caplog: the package logger does not propagate to the root handler.
  messages: list[str] = []

  class _Collect(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
      messages.append(record.getMessage())

  logger = logging.getLogger("birdnet.utils.downloads")
  handler = _Collect()
  previous_level = logger.level
  logger.setLevel(logging.INFO)
  logger.addHandler(handler)
  try:
    _download([(None, tmp_path / "models" / "model-fp32")])
  finally:
    logger.removeHandler(handler)
    logger.setLevel(previous_level)

  assert capsys.readouterr().out == ""
  assert messages == ["Extracting a model..."]
