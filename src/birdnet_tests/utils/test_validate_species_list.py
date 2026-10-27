import csv
from pathlib import Path

import pytest

from birdnet.utils.species_lists import validate_species_list
from birdnet_tests.test_files import TEST_FILES_DIR

pytestmark = pytest.mark.no_tf


def _write_raven_labels(path: Path, rows: list[tuple[str, str]]) -> Path:
  with path.open("w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerows(rows)
  return path


def test_a_plain_file_is_read_line_by_line(tmp_path: Path) -> None:
  species = tmp_path / "labels.txt"
  species.write_text("Sci a_Common a\nSci b_Common b\n", encoding="utf-8")

  assert list(validate_species_list(species)) == ["Sci a_Common a", "Sci b_Common b"]


def test_a_raven_label_file_drops_the_id_column(tmp_path: Path) -> None:
  labels = _write_raven_labels(
    tmp_path / "label_names.csv",
    [("Scia1", "Sci a_Common a"), ("Scib2", "Sci b_Common b")],
  )

  assert list(validate_species_list(labels)) == ["Sci a_Common a", "Sci b_Common b"]


def test_a_raven_label_with_a_comma_stays_whole(tmp_path: Path) -> None:
  labels = _write_raven_labels(
    tmp_path / "label_names.csv", [("Scia1", "Sci a_Common, the")]
  )

  assert list(validate_species_list(labels)) == ["Sci a_Common, the"]


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_a_raven_label_file_in_either_encoding_the_analyzer_writes(
  tmp_path: Path, encoding: str
) -> None:
  """On Windows the analyzer writes the file in the locale's encoding."""
  labels = tmp_path / "label_names.csv"
  labels.write_bytes("Sylv1,Sylvia atricapilla_Mönchsgrasmücke\r\n".encode(encoding))

  assert list(validate_species_list(labels)) == ["Sylvia atricapilla_Mönchsgrasmücke"]


def test_a_raven_label_file_without_an_id_column_is_rejected(tmp_path: Path) -> None:
  labels = tmp_path / "label_names.csv"
  labels.write_text("Sci a_Common a\nSci b_Common b\n", encoding="utf-8")

  with pytest.raises(ValueError, match=r"Failed to read species list") as err:
    validate_species_list(labels)
  assert "not a Raven label file" in str(err.value.__cause__)


def test_the_analyzer_raven_fixture() -> None:
  labels = (
    TEST_FILES_DIR / "custom_models" / "raven" / "CustomClassifier" / "labels"
  ) / "label_names.csv"

  species = list(validate_species_list(labels))
  assert species[:2] == [
    "Cardinalis cardinalis_Northern Cardinal",
    "Cyanocitta cristata_Blue Jay",
  ]
  assert not any("," in name for name in species)
