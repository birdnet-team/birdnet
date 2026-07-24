"""Validation error paths of ``load`` / ``load_custom`` for acoustic models.

These exercise the argument validation that happens *before* any model is
downloaded or loaded, so they run without the ``load_model`` marker.
"""

import re
from pathlib import Path

import pytest

from birdnet.model_loader import load, load_custom

# ----------------------------- shared top-level validators -----------------------


def test_unknown_model_type_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unknown model type: foo\."):
    load("foo", "2.4", "tf")  # type: ignore[arg-type]


def test_unknown_backend_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unknown model backend: zzz\."):
    load("acoustic", "2.4", "zzz")  # type: ignore[arg-type]


def test_unknown_precision_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unsupported model precision: fp8\."):
    load("acoustic", "2.4", "tf", precision="fp8")  # type: ignore[arg-type]


def test_unknown_acoustic_version_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unsupported model version: 9\.9\."):
    load("acoustic", "9.9", "tf")  # type: ignore[arg-type]


def test_unknown_language_v2_4_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Language 'zz' is not supported by model v2\.4\."
  ):
    load("acoustic", "2.4", "tf", lang="zz")


def test_unknown_language_v3_0_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Language 'zz' is not supported by model v3\.0\."
  ):
    load("acoustic", "3.0", "tf", lang="zz")


def test_unknown_library_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unsupported TensorFlow library: zzz\."):
    load("acoustic", "2.4", "tf", library="zzz")  # type: ignore[arg-type]


def test_litert_library_unavailable_raises_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr("birdnet.model_loader.litert_installed", lambda: False)

  with pytest.raises(ValueError, match=r"Install birdnet with \[litert\] option\."):
    load("acoustic", "2.4", "tf", library="litert")


# ----------------------------- load() precision guards ---------------------------


def test_v2_4_pb_non_fp32_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic pb model: fp16\."
  ):
    load("acoustic", "2.4", "pb", precision="fp16")


def test_v3_0_tf_int8_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic tf model: int8\."
  ):
    load("acoustic", "3.0", "tf", precision="int8")


def test_v3_0_pb_non_fp32_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic pb model: fp16\."
  ):
    load("acoustic", "3.0", "pb", precision="fp16")


def test_v3_0_pt_non_fp32_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic pt model: fp16\."
  ):
    load("acoustic", "3.0", "pt", precision="fp16")


def test_v3_0_onnx_int8_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic onnx model: int8\."
  ):
    load("acoustic", "3.0", "onnx", precision="int8")


# ----------------------------- load() unsupported backend ------------------------


def test_v2_4_pt_backend_unsupported_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'pt' for acoustic model v2\.4\."
  ):
    load("acoustic", "2.4", "pt")


def test_v2_4_onnx_backend_unsupported_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'onnx' for acoustic model v2\.4\."
  ):
    load("acoustic", "2.4", "onnx")


# ----------------------------- load_custom() path validators ---------------------


def test_custom_missing_model_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(ValueError, match=r"does not exist!"):
    load_custom("acoustic", "2.4", "tf", tmp_path / "nope.tflite", species)


def test_custom_missing_species_list_raises_error(tmp_path: Path) -> None:
  model = tmp_path / "m.tflite"
  model.write_bytes(b"x")
  with pytest.raises(ValueError, match=r"Species list file .* does not exist!"):
    load_custom("acoustic", "2.4", "tf", model, tmp_path / "nope.txt")


def test_custom_tf_directory_is_not_a_file_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  a_dir = tmp_path / "a_dir"
  a_dir.mkdir()
  with pytest.raises(ValueError, match=r"Model file .* does not exist!"):
    load_custom("acoustic", "2.4", "tf", a_dir, species)


def test_custom_tf_wrong_suffix_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  model = tmp_path / "m.bin"
  model.write_bytes(b"x")
  with pytest.raises(ValueError, match=r"is not a valid TFLite model file!"):
    load_custom("acoustic", "2.4", "tf", model, species)


def test_custom_pb_not_a_directory_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  model = tmp_path / "m.tflite"
  model.write_bytes(b"x")
  with pytest.raises(ValueError, match=r"Model folder .* does not exist!"):
    load_custom("acoustic", "2.4", "pb", model, species)


def test_custom_pb_missing_protobuf_files_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  empty = tmp_path / "empty"
  empty.mkdir()
  with pytest.raises(
    ValueError, match=r"does not contain valid protobuf model files!"
  ):
    load_custom("acoustic", "2.4", "pb", empty, species)


def test_custom_pt_directory_is_not_a_file_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  a_dir = tmp_path / "a_dir"
  a_dir.mkdir()
  with pytest.raises(ValueError, match=r"Model file .* does not exist!"):
    load_custom("acoustic", "3.0", "pt", a_dir, species)


def test_custom_onnx_directory_is_not_a_file_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  a_dir = tmp_path / "a_dir"
  a_dir.mkdir()
  with pytest.raises(ValueError, match=r"Model file .* does not exist!"):
    load_custom("acoustic", "3.0", "onnx", a_dir, species)


# ----------------------------- load_custom() precision guards --------------------


def _pb_dir(tmp_path: Path) -> Path:
  model_path = tmp_path / "pb"
  variables = model_path / "variables"
  variables.mkdir(parents=True)
  (model_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")
  (variables / "variables.index").write_bytes(b"index")
  return model_path


def test_custom_v2_4_pb_non_fp32_precision_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic pb model: fp16\."
  ):
    load_custom("acoustic", "2.4", "pb", _pb_dir(tmp_path), species, precision="fp16")


def test_custom_v3_0_tf_int8_precision_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  model = tmp_path / "m.tflite"
  model.write_bytes(b"x")
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic tf model: int8\."
  ):
    load_custom("acoustic", "3.0", "tf", model, species, precision="int8")


def test_custom_v3_0_pb_non_fp32_precision_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic pb model: fp16\."
  ):
    load_custom("acoustic", "3.0", "pb", tmp_path, species, precision="fp16")


def test_custom_v3_0_pt_non_fp32_precision_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic pt model: fp16\."
  ):
    load_custom("acoustic", "3.0", "pt", tmp_path, species, precision="fp16")


def test_custom_v3_0_onnx_int8_precision_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for acoustic onnx model: int8\."
  ):
    load_custom("acoustic", "3.0", "onnx", tmp_path, species, precision="int8")


# ----------------------------- load_custom() unsupported / kwargs ----------------


def test_custom_v2_4_pt_backend_unsupported_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  model = tmp_path / "m.pt"
  model.write_bytes(b"x")
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'pt' for acoustic model v2\.4\."
  ):
    load_custom("acoustic", "2.4", "pt", model, species)


def test_custom_v2_4_tf_unknown_classifier_type_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  model = tmp_path / "m.tflite"
  model.write_bytes(b"x")
  with pytest.raises(ValueError, match=r"Unknown classifier type: 'bad'\."):
    load_custom(
      "acoustic",
      "2.4",
      "tf",
      model,
      species,
      check_validity=False,
      classifier_type="bad",
    )


def test_custom_v2_4_pb_non_bool_is_raven_raises_error(tmp_path: Path) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(
    ValueError,
    match=re.escape("Parameter 'is_raven' must be of type bool"),
  ):
    load_custom(
      "acoustic", "2.4", "pb", _pb_dir(tmp_path), species, is_raven="yes"
    )
