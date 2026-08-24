import csv
from pathlib import Path

import pytest

import birdnet.acoustic.models.v3_0.model as acoustic_model
import birdnet.geo.models.v3_0.model as geo_model
import birdnet.utils.label_manifest as label_manifest
import birdnet.utils.taxonomy_v3 as taxonomy_v3
from birdnet.utils.label_manifest import sha256_bytes


def test_taxonomy_v3_available_checks_the_content(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  taxonomy_path = tmp_path / "taxonomy.csv"

  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_PATH", taxonomy_path)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SIZE", 3)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SHA256", sha256_bytes(b"abc"))

  assert not taxonomy_v3.taxonomy_v3_available()

  taxonomy_path.write_bytes(b"ab")
  assert not taxonomy_v3.taxonomy_v3_available()

  # Right length, another release's bytes: this is what size alone let through.
  taxonomy_path.write_bytes(b"abd")
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
  # The download is checksum-verified, so the fixture has to pin the digest of
  # what the fake download writes.
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SHA256", sha256_bytes(b"data"))
  monkeypatch.setattr(taxonomy_v3, "_LEGACY_TAXONOMY_V3_PATH", tmp_path / "legacy.csv")

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

  monkeypatch.setattr(label_manifest, "download_file_tqdm", fake_download)

  result = taxonomy_v3.ensure_taxonomy_v3_available()

  assert result == taxonomy_path
  assert taxonomy_path.read_bytes() == b"data"
  assert not lock_dir.exists()
  assert calls == [
    (
      "https://github.com/birdnet-team/geomodel/raw/refs/tags/v3.0.4/taxonomy_v0.2-Jun2026.csv",
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


@pytest.mark.load_model
def test_every_mapped_language_column_exists_in_the_taxonomy(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A language mapped to a column the taxonomy does not have fails silently.

  `_generate_lang_files` falls back to the English name per species, so such a
  language yields a complete, plausible file of entirely English names - which
  is how Estonian survived a taxonomy that had already dropped it.
  """
  taxonomy_path = tmp_path / "taxonomy.csv"
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_PATH", taxonomy_path)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_LOCK_DIR", tmp_path / ".taxonomy.lock")

  taxonomy_v3.ensure_taxonomy_v3_available()
  with open(taxonomy_path, encoding="utf-8", newline="") as f:
    columns = set(next(csv.reader(f)))

  assert set(geo_model._LANGUAGE_TO_COLUMN.values()) <= columns
  assert set(acoustic_model._LANGUAGE_TO_COLUMN.values()) <= columns
