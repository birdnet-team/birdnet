from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import birdnet.acoustic.models.v3_0.model as model_module
from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.globals import ENV_VAR_APP_DATA, VALID_MODEL_LANGUAGES_V3_0
from birdnet.utils.label_manifest import sha256_bytes

pytestmark = pytest.mark.no_tf

if TYPE_CHECKING:
  from pytest import MonkeyPatch


class ProbeDownloader(AcousticDownloaderBaseV3_0):
  lang_dir: Path

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return cls.lang_dir


def _point_at(monkeypatch: MonkeyPatch, app_data: Path, labels: bytes) -> None:
  """Give the generator a private app data directory holding this label file.

  The file is written where the module looks for it, a path named after the
  digest pinned here.
  """
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(app_data))
  monkeypatch.setattr(model_module, "_LABELS_DL_SIZE", len(labels))
  monkeypatch.setattr(model_module, "_LABELS_DL_SHA256", sha256_bytes(labels))
  installed = model_module._labels_raw_path()
  installed.parent.mkdir(parents=True, exist_ok=True)
  installed.write_bytes(labels)
  ProbeDownloader.lang_dir = app_data / "labels"


def _generated(tmp_path: Path, monkeypatch: MonkeyPatch, labels: bytes) -> Path:
  _point_at(monkeypatch, tmp_path, labels)
  ProbeDownloader._generate_lang_files()
  return ProbeDownloader.lang_dir


def _english_labels(tmp_path: Path, monkeypatch: MonkeyPatch, labels: bytes) -> str:
  lang_dir = _generated(tmp_path, monkeypatch, labels)
  return (lang_dir / "en_us.txt").read_text(encoding="utf-8")


def test_generate_lang_files_from_the_label_file(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  lang_dir = _generated(
    tmp_path,
    monkeypatch,
    "ix,birdnet_id,scientific_name,common_name,common_name_de,common_name_zh-CN\n"
    "0,BN1,Aaa aaa,English One,Deutsch Eins,中文一\n"
    "1,BN2,Bbb bbb,English Two,,\n".encode(),
  )

  assert list(ProbeDownloader.AVAILABLE_LANGUAGES) == VALID_MODEL_LANGUAGES_V3_0
  assert (lang_dir / "en_us.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_English Two"
  )
  assert (lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_Deutsch Eins\nBbb bbb_English Two"
  )
  assert (lang_dir / "zh.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_中文一\nBbb bbb_English Two"
  )
  # A language the file has no column for is English throughout.
  assert (lang_dir / "fr.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_English Two"
  )


def test_a_class_without_a_common_name_keeps_its_position(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  # Labels pair with the model's outputs by position, so a class the label
  # file gives no common name must still take up its line.
  lang_dir = _generated(
    tmp_path,
    monkeypatch,
    b"scientific_name,common_name,common_name_de\n"
    b"Aaa aaa,English One,\n"
    b"Bbb bbb,,Deutsch Zwei\n"
    b"Ccc ccc,English Three,\n",
  )

  assert (lang_dir / "en_us.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_Bbb bbb\nCcc ccc_English Three"
  )
  assert (lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_Deutsch Zwei\nCcc ccc_English Three"
  )


@pytest.mark.parametrize(
  ("row", "label"),
  [
    (b",Sound_(event)", "Sound (event)_Sound (event)"),
    (b"Bbb bbb", "Bbb bbb_Bbb bbb"),
  ],
  ids=["no scientific name", "fewer fields than the header"],
)
def test_a_class_with_one_name_only_keeps_its_position(
  tmp_path: Path, monkeypatch: MonkeyPatch, row: bytes, label: str
) -> None:
  labels = (
    b"scientific_name,common_name\nAaa aaa,English One\n"
    + row
    + b"\nCcc ccc,English Three\n"
  )

  assert _english_labels(tmp_path, monkeypatch, labels) == (
    f"Aaa aaa_English One\n{label}\nCcc ccc_English Three"
  )


def test_a_row_without_any_name_is_no_class(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  lang_dir = _generated(
    tmp_path,
    monkeypatch,
    b"scientific_name,common_name,common_name_de\n"
    b"Aaa aaa,English One,Deutsch Eins\n"
    b",,\n"
    b"Bbb bbb,English Two,Deutsch Zwei\n",
  )

  assert (lang_dir / "en_us.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_English Two"
  )
  # The class after the skipped row still gets the names of its own row.
  assert (lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_Deutsch Eins\nBbb bbb_Deutsch Zwei"
  )


def test_a_row_longer_than_the_header_keeps_its_names(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  labels = b"scientific_name,common_name\nAaa aaa,English One,stray,cells\n"

  assert _english_labels(tmp_path, monkeypatch, labels) == "Aaa aaa_English One"


def test_a_byte_order_mark_does_not_hide_the_first_column(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  labels = b"\xef\xbb\xbfscientific_name,common_name\nAaa aaa,English One\n"

  assert _english_labels(tmp_path, monkeypatch, labels) == "Aaa aaa_English One"


def test_a_name_with_an_underscore_does_not_move_the_label_separator(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  lang_dir = _generated(
    tmp_path,
    monkeypatch,
    b"scientific_name,common_name,common_name_de\n"
    b"Zipper_(clothing),Zipper_(clothing),\n"
    b"Aaa sp.nov.Bbb_Ccc,English_One,Deutsch_Eins\n",
  )

  en_us = (lang_dir / "en_us.txt").read_text(encoding="utf-8")
  assert en_us == (
    "Zipper (clothing)_Zipper (clothing)\nAaa sp.nov.Bbb Ccc_English One"
  )
  assert [label.split("_", 1)[0] for label in en_us.split("\n")] == [
    "Zipper (clothing)",
    "Aaa sp.nov.Bbb Ccc",
  ]
  assert (lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Zipper (clothing)_Zipper (clothing)\nAaa sp.nov.Bbb Ccc_Deutsch Eins"
  )


def test_a_label_file_in_another_layout_is_refused(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  # Read as it is, such a file would give no class at all.
  _point_at(monkeypatch, tmp_path, b"idx;id;sci_name;com_name\n0;x;Aaa aaa;One\n")

  with pytest.raises(ValueError, match="no column 'scientific_name'"):
    ProbeDownloader._generate_lang_files()


def test_a_label_file_without_english_names_is_refused(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  # Read as it is, every class would carry its scientific name as its English one.
  _point_at(monkeypatch, tmp_path, b"scientific_name,common_name_de\nAaa aaa,Eins\n")

  with pytest.raises(ValueError, match="no column 'common_name'"):
    ProbeDownloader._generate_lang_files()


def test_ensure_labels_regenerates_for_another_label_file(
  tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
  header = b"scientific_name,common_name,common_name_de\n"
  _point_at(monkeypatch, tmp_path, header + b"Aaa aaa,English One,Alter Name\n")

  ProbeDownloader.ensure_labels_available()
  de_file = ProbeDownloader.lang_dir / "de.txt"
  assert de_file.read_text(encoding="utf-8") == "Aaa aaa_Alter Name"

  # A new label release: other content, so another pin and another path.
  _point_at(monkeypatch, tmp_path, header + b"Aaa aaa,English One,Neuer Name\n")

  ProbeDownloader.ensure_labels_available()

  assert de_file.read_text(encoding="utf-8") == "Aaa aaa_Neuer Name"


def test_the_labels_need_no_taxonomy(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
  # Offline, as every test without load_model is: fetching the taxonomy raises.
  _point_at(
    monkeypatch, tmp_path, b"scientific_name,common_name\nAaa aaa,English One\n"
  )

  ProbeDownloader.ensure_labels_available()

  assert ProbeDownloader._check_labels_available()
  assert not (tmp_path / "taxonomy-v3").exists()
