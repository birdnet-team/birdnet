from pathlib import Path

import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTBackendFP32V3_0
from birdnet.model_loader import load


def test_v3_0_tf_backend_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unsupported backend 'tf' for acoustic model v3.0.",
  ):
    load("acoustic", "3.0", "tf")  # type: ignore[arg-type]


def test_v3_0_pt_with_library_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    load("acoustic", "3.0", "pt", precision="fp32", library="tflite")  # type: ignore[call-arg]


def test_v3_0_pt_runtime_missing_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.torch_installed", lambda: False)

  with pytest.raises(
    ValueError,
    match=r"Install birdnet with \[pt\] option.",
  ):
    load("acoustic", "3.0", "pt")


def test_v3_0_onnx_runtime_missing_raises_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: False)

  with pytest.raises(
    ValueError,
    match=r"Install birdnet with \[onnx\] option.",
  ):
    load("acoustic", "3.0", "onnx")


def test_v3_0_pt_type_is_correct(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.torch_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticPTDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (Path("birdnet_v3.pt"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "pt")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticPTBackendFP32V3_0


def test_v3_0_onnx_type_is_correct(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticOnnxDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (Path("birdnet_v3.onnx"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "onnx")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticOnnxBackendFP32V3_0
