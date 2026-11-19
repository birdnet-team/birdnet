"""A regional language without a name for a class shows its base language's.

The acoustic 3.0 label file leaves a regional column (``pt_PT``, ``es_EC``, the
``zh-CN`` one) empty where only the base language names the class. Generated
here from a small label file through the downloader itself: the regional name
wins, then the base language's, then the English one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import birdnet.acoustic.models.v3_0.model as acoustic_model
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils.label_manifest import sha256_bytes

pytestmark = pytest.mark.no_tf


def _generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, labels: bytes) -> Path:
  """Generate the label files from `labels`; returns their directory."""
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  monkeypatch.setattr(acoustic_model, "_LABELS_DL_SIZE", len(labels))
  monkeypatch.setattr(acoustic_model, "_LABELS_DL_SHA256", sha256_bytes(labels))
  raw = acoustic_model._labels_raw_path()
  raw.parent.mkdir(parents=True)
  raw.write_bytes(labels)
  lang_dir = tmp_path / "labels"

  class _Downloader(acoustic_model.AcousticDownloaderBaseV3_0):
    @classmethod
    def _get_lang_dir(cls) -> Path:
      return lang_dir

  _Downloader._generate_lang_files()
  return lang_dir


def _lines(lang_dir: Path, lang: str) -> list[str]:
  return (lang_dir / f"{lang}.txt").read_text(encoding="utf-8").split("\n")


@pytest.mark.parametrize(
  ("lang", "base_column"), sorted(acoustic_model._FALLBACK_COLUMN.items())
)
def test_a_regional_language_reads_its_base_column_before_english(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lang: str, base_column: str
) -> None:
  own_column = acoustic_model._LANGUAGE_TO_COLUMN[lang]
  labels = (
    f"scientific_name,common_name,{own_column},{base_column}\n"
    "Scivia one,Robin,Regional robin,Base robin\n"
    "Scivia two,Sparrow,,Base sparrow\n"
    "Scivia three,Eagle,,\n"
  ).encode()

  lang_dir = _generate(tmp_path, monkeypatch, labels)

  assert _lines(lang_dir, lang) == [
    "Scivia one_Regional robin",
    "Scivia two_Base sparrow",
    "Scivia three_Eagle",
  ]


def test_a_base_language_does_not_read_a_regional_column(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  labels = (
    b"scientific_name,common_name,common_name_pt,common_name_pt_PT\n"
    b"Scivia one,Robin,,Pisco-de-peito-ruivo\n"
  )

  lang_dir = _generate(tmp_path, monkeypatch, labels)

  assert _lines(lang_dir, "pt") == ["Scivia one_Robin"]


def test_the_regional_languages_fall_back_to_their_base_language() -> None:
  """Written out, because the cases above are generated from the table itself."""
  assert acoustic_model._FALLBACK_COLUMN == {
    "es_ec": "common_name_es",
    "es_es": "common_name_es",
    "es_mx": "common_name_es",
    "pt_pt": "common_name_pt",
    "zh": "common_name_zh",
  }
  assert set(acoustic_model._FALLBACK_COLUMN) <= set(acoustic_model._LANGUAGE_TO_COLUMN)
