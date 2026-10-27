"""Species list parsing from label files and Raven label exports."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ordered_set import OrderedSet


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
