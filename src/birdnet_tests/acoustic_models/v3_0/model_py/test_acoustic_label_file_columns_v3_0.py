import csv

import pytest

import birdnet.acoustic.models.v3_0.model as acoustic_model
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0

# The two languages whose column is not named after them.
_OWN_COLUMN = {"en_us": "common_name", "zh": "common_name_zh-CN"}


def _column_named_after(lang: str) -> str:
  language, _, region = lang.partition("_")
  return f"common_name_{language}" + (f"_{region.upper()}" if region else "")


@pytest.mark.no_tf
@pytest.mark.parametrize("lang", sorted(acoustic_model._LANGUAGE_TO_COLUMN))
def test_a_language_reads_its_own_column(lang: str) -> None:
  """The label file has neighbouring columns (``pt`` beside ``pt_PT``, ``zh``
  beside ``zh-CN``), so a wrong one still yields a complete file of names."""
  expected = _OWN_COLUMN.get(lang, _column_named_after(lang))

  assert acoustic_model._LANGUAGE_TO_COLUMN[lang] == expected


@pytest.mark.load_model
@pytest.mark.no_tf
def test_every_mapped_language_column_exists_in_the_label_file() -> None:
  """A language mapped to a column the label file does not have fails silently.

  `_generate_lang_files` falls back to the English name per class, so such a
  language yields a complete, plausible file of entirely English names.
  """
  AcousticOnnxDownloaderV3_0.ensure_labels_available()

  labels_path = acoustic_model._labels_raw_path()
  with open(labels_path, encoding="utf-8-sig", newline="") as f:
    columns = set(next(csv.reader(f)))

  assert set(acoustic_model._LANGUAGE_TO_COLUMN.values()) <= columns


@pytest.mark.load_model
@pytest.mark.no_tf
def test_the_label_file_has_one_row_per_class() -> None:
  AcousticOnnxDownloaderV3_0.ensure_labels_available()

  with open(acoustic_model._labels_raw_path(), encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

  assert len(rows) == acoustic_model.AcousticModelV3_0.get_n_classes()
  # Labels pair with the model's outputs by position.
  assert [row["ix"] for row in rows] == [str(position) for position in range(len(rows))]
