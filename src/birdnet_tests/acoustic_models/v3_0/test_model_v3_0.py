from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import birdnet.acoustic.models.v3_0.model as model_module
from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.globals import VALID_MODEL_LANGUAGES_V3_0
from birdnet.utils.label_manifest import LabelInput, sha256_file

if TYPE_CHECKING:
  from pytest import MonkeyPatch


class ProbeDownloader(AcousticDownloaderBaseV3_0):
  lang_dir: Path

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return cls.lang_dir


def _point_at(monkeypatch: MonkeyPatch, labels_raw: Path, taxonomy: Path) -> None:
  """Redirect every artifact the generator touches into the test's directory.

  Missing one is not obvious: the manifest stats the paths its `LabelInput`s
  name, so an unpatched one silently reaches into the real app data directory
  and the test then passes only on a machine that happens to have a cache.
  """
  monkeypatch.setattr(model_module, "_LABELS_RAW_PATH", labels_raw)
  monkeypatch.setattr(model_module, "_LABELS_DL_SIZE", labels_raw.stat().st_size)
  monkeypatch.setattr(model_module, "_LABELS_DL_SHA256", sha256_file(labels_raw))
  monkeypatch.setattr(
    model_module, "_LEGACY_LABELS_RAW_PATH", labels_raw.parent / "legacy.csv"
  )
  monkeypatch.setattr(model_module, "_SETUP_LOCK_DIR", labels_raw.parent / ".lock")
  # Recomputed per call, the way the pinned constants track the current release:
  # rewriting the file is then exactly a new taxonomy release.
  monkeypatch.setattr(
    model_module,
    "get_taxonomy_v3_input",
    lambda: LabelInput(
      path=taxonomy,
      url="https://example.test/taxonomy.csv",
      size=taxonomy.stat().st_size,
      sha256=sha256_file(taxonomy),
    ),
  )
  monkeypatch.setattr(model_module, "ensure_taxonomy_v3_available", lambda: taxonomy)


def test_generate_lang_files_from_taxonomy(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  labels_raw = tmp_path / "labels_raw.csv"
  labels_raw.write_text(
    "sci_name;com_name\nAaa aaa;English One\nBbb bbb;English Two\n",
    encoding="utf-8",
  )
  taxonomy = tmp_path / "taxonomy.csv"
  taxonomy.write_text(
    "sci_name,com_name,common_name_de,common_name_zh-CN\n"
    "Aaa aaa,English One,Deutsch Eins,中文一\n",
    encoding="utf-8",
  )

  _point_at(monkeypatch, labels_raw, taxonomy)
  ProbeDownloader.lang_dir = tmp_path / "labels"

  ProbeDownloader._generate_lang_files()

  assert list(ProbeDownloader.AVAILABLE_LANGUAGES) == VALID_MODEL_LANGUAGES_V3_0
  assert (ProbeDownloader.lang_dir / "en_us.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_English Two"
  )
  assert (ProbeDownloader.lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_Deutsch Eins\nBbb bbb_English Two"
  )
  assert (ProbeDownloader.lang_dir / "zh.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_中文一\nBbb bbb_English Two"
  )


def test_ensure_labels_regenerates_when_only_the_taxonomy_changed(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  """The taxonomy is shared, so it can already be current when we are not.

  Whichever v3.0 model downloads a new taxonomy first makes it "available" for
  every other model and backend. Their label files still hold the names built
  from the previous taxonomy, and neither the raw labels nor the file count
  changes with a taxonomy-only bump - so nothing but the marker notices.
  """
  labels_raw = tmp_path / "labels_raw.csv"
  labels_raw.write_bytes(
    b"sci_name;com_name\nAaa aaa;English One\n",
  )
  taxonomy = tmp_path / "taxonomy.csv"
  taxonomy.write_text(
    "sci_name,com_name,common_name_de\nAaa aaa,English One,Alter Name\n",
    encoding="utf-8",
  )

  # The taxonomy is present the whole time: another model fetched it already.
  _point_at(monkeypatch, labels_raw, taxonomy)
  ProbeDownloader.lang_dir = tmp_path / "labels"

  ProbeDownloader.ensure_labels_available()
  de_file = ProbeDownloader.lang_dir / "de.txt"
  assert de_file.read_text(encoding="utf-8") == "Aaa aaa_Alter Name"

  # A new taxonomy release: same on-disk path, different content and identity.
  taxonomy.write_text(
    "sci_name,com_name,common_name_de\nAaa aaa,English One,Neuer Name\n",
    encoding="utf-8",
  )

  ProbeDownloader.ensure_labels_available()

  assert de_file.read_text(encoding="utf-8") == "Aaa aaa_Neuer Name"
