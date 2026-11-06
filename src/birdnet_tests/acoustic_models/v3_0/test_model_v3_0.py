from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import birdnet.acoustic.models.v3_0.model as model_module
from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.globals import ENV_VAR_APP_DATA, VALID_MODEL_LANGUAGES_V3_0
from birdnet.utils.label_manifest import LabelInput, sha256_file

pytestmark = pytest.mark.no_tf

if TYPE_CHECKING:
  from pytest import MonkeyPatch


class ProbeDownloader(AcousticDownloaderBaseV3_0):
  lang_dir: Path

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return cls.lang_dir


def _point_at(
  monkeypatch: MonkeyPatch, app_data: Path, labels_raw: Path, taxonomy: Path
) -> None:
  """Give the generator a private app data directory holding these inputs.

  `labels_raw` is moved to where the module looks for it, a path named after
  the digest pinned here.
  """
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(app_data))
  monkeypatch.setattr(model_module, "_LABELS_DL_SIZE", labels_raw.stat().st_size)
  monkeypatch.setattr(model_module, "_LABELS_DL_SHA256", sha256_file(labels_raw))
  installed = model_module._labels_raw_path()
  installed.parent.mkdir(parents=True)
  labels_raw.replace(installed)
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

  _point_at(monkeypatch, tmp_path, labels_raw, taxonomy)
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


def test_a_class_without_a_common_name_keeps_its_position(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  # Labels pair with the model's outputs by position, so a class the label
  # file gives no common name must still take up its line.
  labels_raw = tmp_path / "labels_raw.csv"
  labels_raw.write_bytes(
    b"sci_name;com_name\nAaa aaa;English One\nBbb bbb;\nCcc ccc;English Three\n",
  )
  taxonomy = tmp_path / "taxonomy.csv"
  taxonomy.write_text(
    "sci_name,com_name,common_name_de\nBbb bbb,,Deutsch Zwei\n",
    encoding="utf-8",
  )

  _point_at(monkeypatch, tmp_path, labels_raw, taxonomy)
  ProbeDownloader.lang_dir = tmp_path / "labels"

  ProbeDownloader._generate_lang_files()

  assert (ProbeDownloader.lang_dir / "en_us.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_Bbb bbb\nCcc ccc_English Three"
  )
  assert (ProbeDownloader.lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_Deutsch Zwei\nCcc ccc_English Three"
  )


def _english_labels(tmp_path: Path, monkeypatch: MonkeyPatch, labels: bytes) -> str:
  """The generated English label file for `labels`, with an empty taxonomy."""
  labels_raw = tmp_path / "labels_raw.csv"
  labels_raw.write_bytes(labels)
  taxonomy = tmp_path / "taxonomy.csv"
  taxonomy.write_text("sci_name,com_name,common_name_de\n", encoding="utf-8")

  _point_at(monkeypatch, tmp_path, labels_raw, taxonomy)
  ProbeDownloader.lang_dir = tmp_path / "labels"

  ProbeDownloader._generate_lang_files()

  return (ProbeDownloader.lang_dir / "en_us.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
  ("row", "label"),
  [
    (b";Sound_(event)", "Sound (event)_Sound (event)"),
    (b"Bbb bbb", "Bbb bbb_Bbb bbb"),
  ],
  ids=["no scientific name", "fewer fields than the header"],
)
def test_a_class_with_one_name_only_keeps_its_position(
  tmp_path: Path, monkeypatch: MonkeyPatch, row: bytes, label: str
) -> None:
  labels = (
    b"sci_name;com_name\nAaa aaa;English One\n" + row + b"\nCcc ccc;English Three\n"
  )

  assert _english_labels(tmp_path, monkeypatch, labels) == (
    f"Aaa aaa_English One\n{label}\nCcc ccc_English Three"
  )


def test_a_row_without_any_name_is_no_class(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  labels = b"sci_name;com_name\nAaa aaa;English One\n;\nBbb bbb;English Two\n"

  assert _english_labels(tmp_path, monkeypatch, labels) == (
    "Aaa aaa_English One\nBbb bbb_English Two"
  )


def test_a_byte_order_mark_does_not_hide_the_first_column(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  labels = b"\xef\xbb\xbfsci_name;com_name\nAaa aaa;English One\n"

  assert _english_labels(tmp_path, monkeypatch, labels) == "Aaa aaa_English One"


def test_a_name_with_an_underscore_does_not_move_the_label_separator(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  labels_raw = tmp_path / "labels_raw.csv"
  labels_raw.write_bytes(
    b"sci_name;com_name\n"
    b"Zipper_(clothing);Zipper_(clothing)\n"
    b"Aaa sp.nov.Bbb_Ccc;English One\n",
  )
  # The taxonomy is still joined on the name as the label file spells it.
  taxonomy = tmp_path / "taxonomy.csv"
  taxonomy.write_text(
    "sci_name,com_name,common_name_de\nAaa sp.nov.Bbb_Ccc,English One,Deutsch Eins\n",
    encoding="utf-8",
  )

  _point_at(monkeypatch, tmp_path, labels_raw, taxonomy)
  ProbeDownloader.lang_dir = tmp_path / "labels"

  ProbeDownloader._generate_lang_files()

  en_us = (ProbeDownloader.lang_dir / "en_us.txt").read_text(encoding="utf-8")
  assert en_us == (
    "Zipper (clothing)_Zipper (clothing)\nAaa sp.nov.Bbb Ccc_English One"
  )
  assert [label.split("_", 1)[0] for label in en_us.split("\n")] == [
    "Zipper (clothing)",
    "Aaa sp.nov.Bbb Ccc",
  ]
  assert (ProbeDownloader.lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Zipper (clothing)_Zipper (clothing)\nAaa sp.nov.Bbb Ccc_Deutsch Eins"
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
  _point_at(monkeypatch, tmp_path, labels_raw, taxonomy)
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
