"""Which error `load` / `load_custom` report when two arguments are wrong at once.

Each call is rejected by two checks; the one the loader runs first must win.
"""

from pathlib import Path

import pytest

from birdnet.model_loader import load, load_custom

pytestmark = pytest.mark.no_tf


@pytest.fixture
def species_list(tmp_path: Path) -> Path:
  path = tmp_path / "labels.txt"
  path.write_text("a\n", encoding="utf-8")
  return path


def _file(tmp_path: Path, name: str) -> Path:
  path = tmp_path / name
  path.write_bytes(b"x")
  return path


def test_load_checks_the_precision_before_keywords() -> None:
  with pytest.raises(ValueError, match=r"^Unsupported model precision for acoustic"):
    load("acoustic", "3.0", "onnx", precision="int8", foo=1)


def test_load_checks_keywords_before_the_library() -> None:
  with pytest.raises(ValueError, match=r"^Unexpected keyword arguments: foo\."):
    load("acoustic", "3.0", "tf", library="bogus", foo=1)


def test_load_checks_keywords_before_the_runtime(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: False)
  with pytest.raises(ValueError, match=r"^Unexpected keyword arguments: foo\."):
    load("acoustic", "3.0", "onnx", foo=1)


def test_load_custom_checks_the_precision_before_the_path(
  tmp_path: Path, species_list: Path
) -> None:
  model = _file(tmp_path, "m.txt")
  with pytest.raises(ValueError, match=r"^Unsupported model precision for acoustic"):
    load_custom("acoustic", "3.0", "onnx", model, species_list, precision="int8")


def test_load_custom_checks_the_path_before_keywords(
  tmp_path: Path, species_list: Path
) -> None:
  model = _file(tmp_path, "m.txt")
  with pytest.raises(ValueError, match=r"is not a valid ONNX model file"):
    load_custom("acoustic", "3.0", "onnx", model, species_list, foo=1)


def test_load_custom_checks_keywords_before_the_library(
  tmp_path: Path, species_list: Path
) -> None:
  model = _file(tmp_path, "m.tflite")
  with pytest.raises(ValueError, match=r"^Unexpected keyword arguments: foo\."):
    load_custom("acoustic", "3.0", "tf", model, species_list, library="bogus", foo=1)


def test_load_custom_checks_the_classifier_type_before_the_v3_0_precision(
  tmp_path: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr("birdnet.model_loader.tf_installed", lambda: True)
  model = _file(tmp_path, "m.tflite")
  with pytest.raises(ValueError, match=r"^Unknown classifier type: 'bad'"):
    load_custom(
      "acoustic",
      "3.0",
      "tf",
      model,
      species_list,
      precision="int8",
      classifier_type="bad",
    )
