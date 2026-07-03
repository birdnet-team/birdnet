from pathlib import Path

import pytest

import birdnet.utils.taxonomy_v3 as taxonomy_v3


def test_taxonomy_v3_available_checks_expected_size(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  taxonomy_path = tmp_path / "taxonomy.csv"
  lock_dir = tmp_path / ".taxonomy.lock"

  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_PATH", taxonomy_path)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_LOCK_DIR", lock_dir)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SIZE", 3)

  assert not taxonomy_v3.taxonomy_v3_available()

  taxonomy_path.write_bytes(b"ab")
  assert not taxonomy_v3.taxonomy_v3_available()

  taxonomy_path.write_bytes(b"abc")
  assert taxonomy_v3.taxonomy_v3_available()


def test_ensure_taxonomy_v3_available_downloads_missing_file(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  taxonomy_path = tmp_path / "taxonomy.csv"
  lock_dir = tmp_path / ".taxonomy.lock"
  calls: list[tuple[str, Path, int | None, str | None]] = []

  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_PATH", taxonomy_path)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_LOCK_DIR", lock_dir)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SIZE", 4)

  def fake_download(
    url: str,
    file_path: Path,
    *,
    download_size: int | None = None,
    description: str | None = None,
  ) -> int:
    calls.append((url, file_path, download_size, description))
    file_path.write_bytes(b"data")
    return 4

  monkeypatch.setattr(taxonomy_v3, "download_file_tqdm", fake_download)

  result = taxonomy_v3.ensure_taxonomy_v3_available()

  assert result == taxonomy_path
  assert taxonomy_path.read_bytes() == b"data"
  assert not lock_dir.exists()
  assert calls == [
    (
      "https://github.com/birdnet-team/geomodel/raw/refs/tags/v3.0.2/taxonomy.csv",
      taxonomy_path,
      4,
      "Downloading shared v3.0 taxonomy",
    )
  ]


@pytest.mark.load_model
def test_ensure_taxonomy_v3_available_downloads_real_file(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  taxonomy_path = tmp_path / "taxonomy.csv"
  lock_dir = tmp_path / ".taxonomy.lock"

  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_PATH", taxonomy_path)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_LOCK_DIR", lock_dir)

  result = taxonomy_v3.ensure_taxonomy_v3_available()

  assert result == taxonomy_path
  assert taxonomy_path.is_file()
  assert not lock_dir.exists()
  assert taxonomy_v3.taxonomy_v3_available()

  with open(taxonomy_path, encoding="utf-8") as f:
    header = f.readline().strip()

  assert "species_code" in header
  assert "sci_name" in header
