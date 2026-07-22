"""Validation error paths of ``load`` / ``load_custom`` for geo models.

These exercise the argument validation that happens *before* any model is
downloaded or loaded, so they run without the ``load_model`` marker.
"""

from pathlib import Path

import pytest

from birdnet.model_loader import load, load_custom


def _species_file(tmp_path: Path) -> Path:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  return species


def _pb_dir(tmp_path: Path) -> Path:
  model_path = tmp_path / "pb"
  variables = model_path / "variables"
  variables.mkdir(parents=True)
  (model_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")
  (variables / "variables.index").write_bytes(b"index")
  return model_path


# ----------------------------- load() version / precision guards -----------------


def test_unknown_geo_version_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unsupported model version: 9\.9\."):
    load("geo", "9.9", "tf")  # type: ignore[arg-type]


def test_v2_4_tf_non_fp32_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo model: fp16\."
  ):
    load("geo", "2.4", "tf", precision="fp16")


def test_v2_4_pb_non_fp32_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo model: fp16\."
  ):
    load("geo", "2.4", "pb", precision="fp16")


def test_v3_0_pb_non_fp32_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo model: fp16\."
  ):
    load("geo", "3.0", "pb", precision="fp16")


def test_v3_0_onnx_int8_precision_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo onnx model: int8\."
  ):
    load("geo", "3.0", "onnx", precision="int8")


# ----------------------------- load() unsupported backend ------------------------


def test_v2_4_pt_backend_unsupported_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'pt' for geo model v2\.4\."
  ):
    load("geo", "2.4", "pt")


def test_v2_4_onnx_backend_unsupported_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'onnx' for geo model v2\.4\."
  ):
    load("geo", "2.4", "onnx")


def test_v3_0_pt_backend_unsupported_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'pt' for geo model v3\.0\."
  ):
    load("geo", "3.0", "pt")


# ----------------------------- load_custom() precision guards --------------------


def test_custom_v2_4_tf_non_fp32_precision_raises_error(tmp_path: Path) -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo model: fp16\."
  ):
    load_custom("geo", "2.4", "tf", tmp_path, _species_file(tmp_path), precision="fp16")


def test_custom_v2_4_pb_non_fp32_precision_raises_error(tmp_path: Path) -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo model: fp16\."
  ):
    load_custom("geo", "2.4", "pb", tmp_path, _species_file(tmp_path), precision="fp16")


def test_custom_v3_0_pb_non_fp32_precision_raises_error(tmp_path: Path) -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo model: fp16\."
  ):
    load_custom("geo", "3.0", "pb", tmp_path, _species_file(tmp_path), precision="fp16")


def test_custom_v3_0_onnx_int8_precision_raises_error(tmp_path: Path) -> None:
  with pytest.raises(
    ValueError, match=r"Unsupported model precision for geo onnx model: int8\."
  ):
    load_custom(
      "geo", "3.0", "onnx", tmp_path, _species_file(tmp_path), precision="int8"
    )


# ----------------------------- load_custom() unsupported backend -----------------


def test_custom_v2_4_pt_backend_unsupported_raises_error(tmp_path: Path) -> None:
  model = tmp_path / "m.pt"
  model.write_bytes(b"x")
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'pt' for geo model v2\.4\."
  ):
    load_custom("geo", "2.4", "pt", model, _species_file(tmp_path))


def test_custom_v3_0_pt_backend_unsupported_raises_error(tmp_path: Path) -> None:
  model = tmp_path / "m.pt"
  model.write_bytes(b"x")
  with pytest.raises(
    ValueError, match=r"Unsupported backend 'pt' for geo model v3\.0\."
  ):
    load_custom("geo", "3.0", "pt", model, _species_file(tmp_path))
