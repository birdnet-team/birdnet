import pytest

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import (
  AcousticOnnxBackendFP16V3_0,
  AcousticOnnxBackendFP32V3_0,
)
from birdnet.acoustic.models.v3_0.pb import AcousticPBBackendFP32V3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTBackendFP32V3_0
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendFP16V3_0,
  AcousticTFBackendFP32V3_0,
)
from birdnet.model_loader import load_custom


def _create_fake_pb_model_dir(tmp_path) -> object:
  model_path = tmp_path / "birdnet_v3_pb"
  variables_dir = model_path / "variables"
  variables_dir.mkdir(parents=True)
  (model_path / "saved_model.pb").write_bytes(b"pb")
  (variables_dir / "variables.data-00000-of-00001").write_bytes(b"data")
  (variables_dir / "variables.index").write_bytes(b"index")
  return model_path


@pytest.mark.no_tf
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


def test_v3_0_custom_pb_type_is_correct(
  tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_path = _create_fake_pb_model_dir(tmp_path)
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded",
    lambda *args, **kwargs: 1,
  )

  model = load_custom("acoustic", "3.0", "pb", model_path, species_list)
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticPBBackendFP32V3_0


def test_v3_0_custom_tf_type_is_correct_fp32(
  tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_path = tmp_path / "birdnet_v3.tflite"
  model_path.write_bytes(b"tf")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  monkeypatch.setattr("birdnet.model_loader.tf_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded",
    lambda *args, **kwargs: 1,
  )

  model = load_custom(
    "acoustic", "3.0", "tf", model_path, species_list, precision="fp32"
  )
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticTFBackendFP32V3_0


def test_v3_0_custom_tf_type_is_correct_fp16(
  tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_path = tmp_path / "birdnet_v3.tflite"
  model_path.write_bytes(b"tf")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  monkeypatch.setattr("birdnet.model_loader.tf_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded",
    lambda *args, **kwargs: 1,
  )

  model = load_custom(
    "acoustic", "3.0", "tf", model_path, species_list, precision="fp16"
  )
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticTFBackendFP16V3_0


@pytest.mark.no_tf
def test_v3_0_custom_onnx_type_is_correct_fp32(
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

  model = load_custom(
    "acoustic", "3.0", "onnx", model_path, species_list, precision="fp32"
  )
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticOnnxBackendFP32V3_0


@pytest.mark.no_tf
def test_v3_0_custom_onnx_type_is_correct_fp16(
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

  model = load_custom(
    "acoustic", "3.0", "onnx", model_path, species_list, precision="fp16"
  )
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticOnnxBackendFP16V3_0


def test_v3_0_custom_pb_with_library_raises_error(tmp_path) -> None:
  model_path = _create_fake_pb_model_dir(tmp_path)
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  with pytest.raises(ValueError, match=r"Unexpected keyword arguments: library."):
    load_custom(
      "acoustic",
      "3.0",
      "pb",
      model_path,
      species_list,
      library="tflite",
    )  # type: ignore[call-arg]


@pytest.mark.no_tf
def test_v3_0_custom_pt_wrong_suffix_raises_error(tmp_path) -> None:
  model_path = tmp_path / "birdnet_v3.bin"
  model_path.write_bytes(b"pt")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  with pytest.raises(ValueError, match=r"is not a valid PT model file"):
    load_custom("acoustic", "3.0", "pt", model_path, species_list)


@pytest.mark.no_tf
def test_v3_0_custom_onnx_wrong_suffix_raises_error(tmp_path) -> None:
  model_path = tmp_path / "birdnet_v3.bin"
  model_path.write_bytes(b"onnx")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")

  with pytest.raises(ValueError, match=r"is not a valid ONNX model file"):
    load_custom("acoustic", "3.0", "onnx", model_path, species_list)
