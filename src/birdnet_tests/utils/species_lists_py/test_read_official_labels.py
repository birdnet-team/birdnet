"""An official label file must name exactly the model's classes, in file order."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from birdnet.utils.species_lists import read_official_labels

pytestmark = pytest.mark.no_tf

_ENGLISH = ["Aa aa_One", "Bb bb_Two", "Cc cc_Three"]


def _write(tmp_path: Path, lines: list[str], name: str = "de.txt") -> Path:
  path = tmp_path / name
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


class _Collect(logging.Handler):
  def __init__(self) -> None:
    super().__init__()
    self.messages: list[str] = []

  def emit(self, record: logging.LogRecord) -> None:
    self.messages.append(record.getMessage())


@pytest.fixture
def warnings() -> Iterator[list[str]]:
  # Not caplog: the package logger does not propagate to the root handler.
  logger = logging.getLogger("birdnet.utils.species_lists")
  handler = _Collect()
  logger.addHandler(handler)
  try:
    yield handler.messages
  finally:
    logger.removeHandler(handler)


def test_returns_the_labels_in_file_order(tmp_path: Path) -> None:
  path = _write(tmp_path, ["B_b", "A_a", "C_c"])

  assert list(read_official_labels(path, 3, "test model")) == ["B_b", "A_a", "C_c"]


def test_the_header_is_required_and_dropped(tmp_path: Path) -> None:
  path = _write(tmp_path, ["header", "A_a", "B_b"])

  labels = read_official_labels(path, 2, "test model", header="header")

  assert list(labels) == ["A_a", "B_b"]


def test_a_missing_header_is_refused(tmp_path: Path) -> None:
  path = _write(tmp_path, ["A_a", "B_b"])

  with pytest.raises(RuntimeError, match=r"does not start with the header 'header'"):
    read_official_labels(path, 2, "test model", header="header")


def test_another_number_of_lines_names_the_file_and_both_counts(
  tmp_path: Path,
) -> None:
  path = _write(tmp_path, ["A_a", "B_b"])

  with pytest.raises(
    RuntimeError, match=r"has 2 lines, but the model outputs 3 classes\."
  ) as info:
    read_official_labels(path, 3, "test model")

  assert str(path) in str(info.value)
  assert str(path.parent) in str(info.value)
  assert "test model" in str(info.value)


def test_extra_lines_are_refused_even_when_they_repeat_earlier_ones(
  tmp_path: Path,
) -> None:
  path = _write(tmp_path, ["A_a", "B_b", "A_a", "C_c"])

  with pytest.raises(RuntimeError, match=r"has 4 lines, but the model outputs 3"):
    read_official_labels(path, 3, "test model")


def test_a_repeated_line_without_an_english_file_is_refused(tmp_path: Path) -> None:
  path = _write(tmp_path, ["A_a", "B_b", "A_a"])

  with pytest.raises(
    RuntimeError,
    match=r"repeats 'A_a' and so names 2 of the model's 3 classes\. The file is "
    r"faulty as shipped and deleting the cache will not help",
  ):
    read_official_labels(path, 3, "test model")


def test_a_repeated_line_takes_the_english_label_at_its_position(
  tmp_path: Path, warnings: list[str]
) -> None:
  """The shape of the 2.4 Polish file: the right number of lines, one of them
  repeating the line before it where the English file names another species."""
  _write(tmp_path, _ENGLISH, name="en_us.txt")
  path = _write(tmp_path, ["Aa aa_Eins", "Bb bb_Zwei", "Bb bb_Zwei"])

  labels = read_official_labels(path, 3, "test model")

  assert list(labels) == ["Aa aa_Eins", "Bb bb_Zwei", "Cc cc_Three"]
  assert len(warnings) == 1
  assert "'Bb bb_Zwei' at line 3" in warnings[0]
  assert "'Cc cc_Three'" in warnings[0]


def test_the_english_file_itself_is_not_repaired(tmp_path: Path) -> None:
  path = _write(tmp_path, ["Aa aa_One", "Bb bb_Two", "Bb bb_Two"], name="en_us.txt")

  with pytest.raises(RuntimeError, match=r"repeats 'Bb bb_Two'"):
    read_official_labels(path, 3, "test model")


def test_an_english_file_of_another_length_does_not_repair(tmp_path: Path) -> None:
  _write(tmp_path, _ENGLISH[:2], name="en_us.txt")
  path = _write(tmp_path, ["Aa aa_Eins", "Bb bb_Zwei", "Bb bb_Zwei"])

  with pytest.raises(RuntimeError, match=r"repeats 'Bb bb_Zwei'"):
    read_official_labels(path, 3, "test model")


def test_a_repeat_the_english_file_cannot_place_is_refused(tmp_path: Path) -> None:
  """Both occurrences carry the English file's scientific name at their
  position, so neither can be told to be the misplaced one."""
  _write(tmp_path, ["Aa aa_One", "Aa aa_Two"], name="en_us.txt")
  path = _write(tmp_path, ["Aa aa_Eins", "Aa aa_Eins"])

  with pytest.raises(RuntimeError, match=r"repeats 'Aa aa_Eins'"):
    read_official_labels(path, 2, "test model")
