"""A TFLite file below a folder whose name is not ASCII.

On Windows the interpreter cannot open such a path itself, and a user profile
named with an umlaut puts the whole app data directory below one.
"""

import shutil
from pathlib import Path

import pytest

import birdnet
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILES_DIR

_MODEL = TEST_FILES_DIR / "custom_models/tf/replace.tflite"
_LABELS = TEST_FILES_DIR / "custom_models/tf/replace_Labels.txt"

# The first name fits the Western European code page, the second does not.
_FOLDERS = ("Jörg Müller", "田中 太郎")


def _n_species_of_copy(tmp_path: Path, folder: str, library: str) -> int:
  model = tmp_path / folder / "model.tflite"
  model.parent.mkdir()
  shutil.copyfile(_MODEL, model)
  return birdnet.inspect_custom_tflite(model, library=library).n_species


def _n_labels() -> int:
  return len(_LABELS.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("folder", _FOLDERS)
def test_tensorflow_opens_it(tmp_path: Path, folder: str) -> None:
  assert _n_species_of_copy(tmp_path, folder, "tflite") == _n_labels()


@pytest.mark.litert
@pytest.mark.parametrize("folder", _FOLDERS)
def test_litert_opens_it(tmp_path: Path, folder: str) -> None:
  ensure_litert_or_skip()
  assert _n_species_of_copy(tmp_path, folder, "litert") == _n_labels()
