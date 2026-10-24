import csv
from pathlib import Path

import pytest

import birdnet.acoustic.models.v3_0.model as acoustic_model
import birdnet.geo.models.v3_0.model as geo_model
import birdnet.utils.taxonomy_v3 as taxonomy_v3
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils import helper
from birdnet.utils.label_manifest import sha256_bytes


def test_taxonomy_v3_available_checks_the_content(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SIZE", 3)
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SHA256", sha256_bytes(b"abc"))
  taxonomy_path = taxonomy_v3._taxonomy_v3_path()
  taxonomy_path.parent.mkdir(parents=True)

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
  # Not created: a first run has to make every directory it writes to.
  app_data = tmp_path / "app-data"
  calls: list[tuple[str, Path, int | None, str | None]] = []

  monkeypatch.setenv(ENV_VAR_APP_DATA, str(app_data))
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SIZE", 4)
  # The download is checksum-verified, so the fixture has to pin the digest of
  # what the fake download writes.
  monkeypatch.setattr(taxonomy_v3, "_TAXONOMY_V3_DL_SHA256", sha256_bytes(b"data"))
  taxonomy_path = taxonomy_v3._taxonomy_v3_path()
  lock_dir = taxonomy_v3._taxonomy_v3_lock_dir()

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

  monkeypatch.setattr(helper, "download_file_tqdm", fake_download)
  assert not app_data.exists()

  result = taxonomy_v3.ensure_taxonomy_v3_available()

  assert result == taxonomy_path
  assert taxonomy_path.is_relative_to(app_data.resolve())
  assert taxonomy_path.read_bytes() == b"data"
  assert not lock_dir.exists()
  assert len(calls) == 1
  url, scratch, download_size, description = calls[0]
  assert (url, download_size, description) == (
    "https://github.com/birdnet-team/geomodel/raw/refs/tags/v3.0.4/taxonomy_v0.2-Jun2026.csv",
    4,
    "Downloading shared v3.0 taxonomy",
  )
  # Into a scratch file beside the target, renamed onto it once verified.
  assert scratch.parent == taxonomy_path.parent
  assert scratch.name.startswith(taxonomy_path.name)
  assert scratch.name.endswith(".unverified")


@pytest.mark.load_model
def test_ensure_taxonomy_v3_available_downloads_real_file(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  taxonomy_path = taxonomy_v3._taxonomy_v3_path()
  lock_dir = taxonomy_v3._taxonomy_v3_lock_dir()

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
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  taxonomy_path = taxonomy_v3._taxonomy_v3_path()

  taxonomy_v3.ensure_taxonomy_v3_available()
  with open(taxonomy_path, encoding="utf-8", newline="") as f:
    columns = set(next(csv.reader(f)))

  assert set(geo_model._LANGUAGE_TO_COLUMN.values()) <= columns
  assert set(acoustic_model._LANGUAGE_TO_COLUMN.values()) <= columns
