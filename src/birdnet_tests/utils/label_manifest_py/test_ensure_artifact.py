from pathlib import Path

import pytest

import birdnet.utils.label_manifest as label_manifest
from birdnet.utils.label_manifest import (
  LabelInput,
  artifact_is_current,
  ensure_artifact,
  sha256_bytes,
  verify_download,
)

_CONTENT = b"the pinned release\n"
_SAME_LENGTH = b"another release!!!\n"


def _expected(tmp_path: Path) -> LabelInput:
  assert len(_SAME_LENGTH) == len(_CONTENT)
  return LabelInput(
    path=tmp_path / "cache" / "artifact.csv",
    url="https://example.org/artifact.csv",
    size=len(_CONTENT),
    sha256=sha256_bytes(_CONTENT),
  )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
  """Records downloads, and fails the test if one happens unexpectedly."""
  calls: list[Path] = []

  def _download(url: str, file_path: Path, **kwargs: object) -> int:
    calls.append(file_path)
    file_path.write_bytes(_CONTENT)
    return len(_CONTENT)

  monkeypatch.setattr(label_manifest, "download_file_tqdm", _download)
  return calls


@pytest.mark.no_tf
def test_a_file_of_the_right_length_but_wrong_content_is_not_current(
  tmp_path: Path,
) -> None:
  """Size is what let one version's taxonomy stand in for another's."""
  expected = _expected(tmp_path)
  expected.path.parent.mkdir()
  expected.path.write_bytes(_SAME_LENGTH)

  assert expected.path.stat().st_size == expected.size
  assert not artifact_is_current(expected)


@pytest.mark.no_tf
def test_replaces_a_file_whose_content_is_from_another_release(
  tmp_path: Path, no_network: list[Path]
) -> None:
  """Detecting the mismatch is not enough - it has to be repaired, or every
  later load regenerates from the same wrong bytes."""
  expected = _expected(tmp_path)
  expected.path.parent.mkdir()
  expected.path.write_bytes(_SAME_LENGTH)

  ensure_artifact(expected, "test artifact")

  assert expected.path.read_bytes() == _CONTENT
  assert no_network == [expected.path]
  assert artifact_is_current(expected)


@pytest.mark.no_tf
def test_does_not_download_when_the_file_is_already_current(
  tmp_path: Path, no_network: list[Path]
) -> None:
  expected = _expected(tmp_path)
  expected.path.parent.mkdir()
  expected.path.write_bytes(_CONTENT)

  ensure_artifact(expected, "test artifact")

  assert no_network == []


@pytest.mark.no_tf
def test_adopts_a_matching_file_from_the_previous_layout(
  tmp_path: Path, no_network: list[Path]
) -> None:
  """This is what keeps the upgrade offline for everyone already holding it."""
  expected = _expected(tmp_path)
  legacy = tmp_path / "legacy.csv"
  legacy.write_bytes(_CONTENT)

  ensure_artifact(expected, "test artifact", legacy_path=legacy)

  assert expected.path.read_bytes() == _CONTENT
  assert no_network == [], "adopting must not download"
  assert not legacy.exists()


@pytest.mark.no_tf
def test_leaves_a_legacy_file_of_another_release_alone(
  tmp_path: Path, no_network: list[Path]
) -> None:
  """It belongs to another installed version still reading it from there."""
  expected = _expected(tmp_path)
  legacy = tmp_path / "legacy.csv"
  legacy.write_bytes(_SAME_LENGTH)

  ensure_artifact(expected, "test artifact", legacy_path=legacy)

  assert legacy.read_bytes() == _SAME_LENGTH
  assert no_network == [expected.path]


@pytest.mark.no_tf
def test_verify_download_discards_a_mismatching_file(tmp_path: Path) -> None:
  expected = _expected(tmp_path)
  expected.path.parent.mkdir()
  expected.path.write_bytes(_SAME_LENGTH)

  with pytest.raises(RuntimeError, match="does not match its expected checksum"):
    verify_download(expected.path, expected)

  assert not expected.path.exists(), "a retry must start from nothing"
