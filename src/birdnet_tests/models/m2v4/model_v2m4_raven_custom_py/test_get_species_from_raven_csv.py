import tempfile
from pathlib import Path

import pytest

from birdnet.models.v2m4.model_v2m4_raven_custom import get_species_from_raven_csv
from birdnet_tests.helper import TEST_FILES_DIR

CLASSIFIER_FOLDER = TEST_FILES_DIR / "v2m4" / "custom_model_raven"


def test_raven_csv_is_loaded():
  path = CLASSIFIER_FOLDER / "CustomClassifier" / "labels" / "label_names.csv"
  species = list(get_species_from_raven_csv(path))
  assert species == [
    "Cardinalis cardinalis_Northern Cardinal",
    "Cyanocitta cristata_Blue Jay",
    "Junco hyemalis_Dark-eyed Junco",
    "Poecile atricapillus_Black-capped Chickadee",
  ]


def test_valid_csv_loads_species_one_species():
  with tempfile.NamedTemporaryFile(prefix="birdnet.test_valid_csv_loads_species_one_species") as tmp_file:
    path = Path(tmp_file.name)
    path.write_text("test1,species1_Species 1", encoding="utf-8")
    species = list(get_species_from_raven_csv(path))
    assert species == [
      "species1_Species 1",
    ]


def test_valid_csv_loads_species_two_species():
  with tempfile.NamedTemporaryFile(prefix="birdnet.test_valid_csv_loads_species_two_species") as tmp_file:
    path = Path(tmp_file.name)
    path.write_text("test1,species1_Species 1\ntest2,species2_Species 2", encoding="utf-8")
    species = list(get_species_from_raven_csv(path))
    assert species == [
      "species1_Species 1",
      "species2_Species 2",
    ]


def test_invalid_csv_raises_exception():
  with tempfile.NamedTemporaryFile(prefix="birdnet.test_invalid_csv_raises_exception") as tmp_file:
    path = Path(tmp_file.name)
    path.write_text("species1_Species 1", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid input format detected! Expected species names in Raven model to be something like 'Card1,Cardinalis cardinalis_Northern Cardinal'."):
      list(get_species_from_raven_csv(path))
