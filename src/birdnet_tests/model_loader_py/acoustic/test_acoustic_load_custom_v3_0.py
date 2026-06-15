import pytest

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTBackendFP32V3_0
from birdnet.model_loader import load_custom


def test_v3_0_custom_pt_type_is_correct(
  tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_path = tmp_path / "birdnet_v3.pt"
  model_path.write_bytes(b"pt")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  monkeypatch.setattr("birdnet.model_loader.torch_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded",
    lambda *args, **kwargs: 1,
  )

  model = load_custom("acoustic", "3.0", "pt", model_path, species_list)
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticPTBackendFP32V3_0


def test_v3_0_custom_onnx_type_is_correct(
  tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_path = tmp_path / "birdnet_v3.onnx"
  model_path.write_bytes(b"onnx")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded",
    lambda *args, **kwargs: 1,
  )

  model = load_custom("acoustic", "3.0", "onnx", model_path, species_list)
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticOnnxBackendFP32V3_0


def test_v3_0_custom_pt_wrong_suffix_raises_error(tmp_path) -> None:
  model_path = tmp_path / "birdnet_v3.bin"
  model_path.write_bytes(b"pt")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  with pytest.raises(ValueError, match=r"is not a valid PT model file"):
    load_custom("acoustic", "3.0", "pt", model_path, species_list)


def test_v3_0_custom_onnx_wrong_suffix_raises_error(tmp_path) -> None:
  model_path = tmp_path / "birdnet_v3.bin"
  model_path.write_bytes(b"onnx")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  with pytest.raises(ValueError, match=r"is not a valid ONNX model file"):
    load_custom("acoustic", "3.0", "onnx", model_path, species_list)
