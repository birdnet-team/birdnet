from pathlib import Path

from birdnet.utils.species_lists import get_species_from_file


def test_get_species_from_file(tmp_path: Path) -> None:
  species_file = tmp_path / "species.txt"
  species_file.write_text("species_a\nspecies_b\nspecies_c\n", encoding="utf-8")
  species = get_species_from_file(species_file)
  assert list(species) == ["species_a", "species_b", "species_c"]
