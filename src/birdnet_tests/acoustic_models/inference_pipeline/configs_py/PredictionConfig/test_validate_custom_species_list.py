from pathlib import Path

import pytest

from birdnet.acoustic.inference.configs import PredictionConfig

MODEL_SPECIES = ["species_a", "species_b", "species_c"]


def test_list_is_valid() -> None:
  result = PredictionConfig.validate_custom_species_list(
    ["species_a", "species_b"], MODEL_SPECIES
  )
  assert result == {"species_a", "species_b"}


def test_set_is_valid() -> None:
  result = PredictionConfig.validate_custom_species_list(
    {"species_a"}, MODEL_SPECIES
  )
  assert result == {"species_a"}


def test_reads_from_file_path(tmp_path: Path) -> None:
  species_file = tmp_path / "species.txt"
  species_file.write_text("species_a\nspecies_c\n", encoding="utf-8")

  result = PredictionConfig.validate_custom_species_list(species_file, MODEL_SPECIES)
  assert result == {"species_a", "species_c"}


def test_reads_from_string_path(tmp_path: Path) -> None:
  species_file = tmp_path / "species.txt"
  species_file.write_text("species_b\n", encoding="utf-8")

  result = PredictionConfig.validate_custom_species_list(
    str(species_file), MODEL_SPECIES
  )
  assert result == {"species_b"}


def test_unknown_species_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"species 'unknown' is not available in the model"
  ):
    PredictionConfig.validate_custom_species_list(["unknown"], MODEL_SPECIES)


def test_non_collection_raises_error() -> None:
  with pytest.raises(TypeError, match=r"custom species list must be a str, path"):
    PredictionConfig.validate_custom_species_list(123, MODEL_SPECIES)  # type: ignore


def test_non_string_element_raises_error() -> None:
  with pytest.raises(TypeError, match=r"custom species list must contain strings"):
    PredictionConfig.validate_custom_species_list([1, 2], MODEL_SPECIES)  # type: ignore
