"""Species list parsing from label files and Raven label exports."""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from pathlib import Path

from ordered_set import OrderedSet

logger = logging.getLogger(__name__)

_ENGLISH_LABELS_NAME = "en_us.txt"


def validate_species_list(species_list: Path) -> OrderedSet[str]:
  loaded_species_list: OrderedSet[str]
  try:
    if species_list.name == RAVEN_LABELS_NAME:
      loaded_species_list = get_species_from_raven_labels(species_list)
    else:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
  except Exception as e:
    raise ValueError(
      f"Failed to read species list from '{species_list.absolute()}'. "
      f"Ensure it is a valid text file."
    ) from e

  if len(loaded_species_list) == 0:
    raise ValueError(f"Species list '{species_list.absolute()}' is empty!")

  return loaded_species_list


def get_species_from_file(
  species_file: Path, /, *, encoding: str = "utf8"
) -> OrderedSet[str]:
  species = OrderedSet(species_file.read_text(encoding).strip().splitlines())
  return species


def _read_label_lines(labels_file: Path) -> list[str]:
  # Split on newlines only: str.splitlines() would also break on U+2028 and
  # friends, silently changing the count if one ever appeared in a label.
  return labels_file.read_text("utf8").strip().split("\n")


def _sci_name(label: str) -> str:
  return label.split("_", 1)[0]


def _repair_repeated_lines(
  labels_file: Path, lines: list[str]
) -> tuple[list[str], list[tuple[int, str, str]]] | None:
  """Replace a repeated line with the English label at its position.

  Only for a language file beside an English one of the same length: the
  scientific names are the same in every language, so the occurrence whose
  scientific name disagrees with the English line is the misplaced one.
  Returns the repaired lines and ``(line number, line, replacement)`` per
  replacement, or None when no replacement can be made.
  """
  english_file = labels_file.with_name(_ENGLISH_LABELS_NAME)
  if labels_file.name == _ENGLISH_LABELS_NAME or not english_file.is_file():
    return None
  english = _read_label_lines(english_file)
  if len(english) != len(lines) or len(set(english)) != len(english):
    return None
  counts = Counter(lines)
  repaired = list(lines)
  replaced: list[tuple[int, str, str]] = []
  for index, (line, english_line) in enumerate(zip(lines, english, strict=True)):
    if counts[line] > 1 and _sci_name(line) != _sci_name(english_line):
      repaired[index] = english_line
      replaced.append((index + 1, line, english_line))
  if not replaced:
    return None
  return repaired, replaced


def read_official_labels(
  labels_file: Path, n_classes: int, model_desc: str, *, header: str | None = None
) -> OrderedSet[str]:
  """Read an official model's label file, which must name exactly its classes.

  The names are paired with the model's outputs by position, so a file with
  another number of lines is refused, and so is one that repeats a line, unless
  the English file beside it supplies the label at the repeated position.
  ``header`` is a first line the file must carry and that is not a label.
  """
  lines = _read_label_lines(labels_file)
  if header is not None:
    if lines[0] != header:
      raise RuntimeError(
        f"The label file '{labels_file}' of the {model_desc} does not start with "
        f"the header '{header}'. Delete '{labels_file.parent}' and load the model "
        "again; if this persists it is a bug in birdnet."
      )
    lines = lines[1:]
  if len(lines) != n_classes:
    raise RuntimeError(
      f"The label file '{labels_file}' of the {model_desc} has {len(lines)} lines, "
      f"but the model outputs {n_classes} classes. Delete '{labels_file.parent}' "
      "and load the model again; if this persists it is a bug in birdnet."
    )
  labels = OrderedSet(lines)
  if len(labels) == n_classes:
    return labels
  repaired = _repair_repeated_lines(labels_file, lines)
  if repaired is not None:
    repaired_lines, replaced = repaired
    labels = OrderedSet(repaired_lines)
    if len(labels) == n_classes:
      for number, line, english_line in replaced:
        logger.warning(
          "Label file '%s' of the %s repeats '%s' at line %d; the English label "
          "'%s' is used there.",
          labels_file,
          model_desc,
          line,
          number,
          english_line,
        )
      return labels
  repeated = ", ".join(
    f"'{line}'" for line, count in Counter(lines).items() if count > 1
  )
  raise RuntimeError(
    f"The label file '{labels_file}' of the {model_desc} repeats {repeated} and "
    f"so names {len(OrderedSet(lines))} of the model's {n_classes} classes. The "
    "file is faulty as shipped and deleting the cache will not help; this is a "
    "bug in birdnet."
  )


# BirdNET-Analyzer's Raven export keeps its species in `labels/label_names.csv`
# as `id,label` rows; the id is derived from the label and is not a species.
RAVEN_LABELS_NAME = "label_names.csv"


def get_species_from_raven_labels(
  labels_file: Path, /, *, encoding: str = "utf8"
) -> OrderedSet[str]:
  # The analyzer writes this file in the locale's encoding, cp1252 on Windows,
  # while its `_Labels.txt` is UTF-8.
  try:
    text = labels_file.read_text(encoding)
  except UnicodeDecodeError:
    text = labels_file.read_text("cp1252")
  rows = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
  if any(len(row) != 2 for row in rows):
    raise ValueError(
      f"'{labels_file}' is not a Raven label file: expected rows of an id and a label."
    )
  return OrderedSet(label for _, label in rows)
