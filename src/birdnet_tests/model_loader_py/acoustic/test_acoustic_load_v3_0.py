from pathlib import Path

import pytest
from ordered_set import OrderedSet
from requests.exceptions import ReadTimeout

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
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_onnxruntime_or_skip, ensure_torch_or_skip


def test_v3_0_pb_with_library_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    load("acoustic", "3.0", "pb", precision="fp32", library="tflite")  # type: ignore[arg-type]


@pytest.mark.load_model
def test_v3_0_pb() -> None:
  try:
    model = load("acoustic", "3.0", "pb", precision="fp32")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_fp32() -> None:
  try:
    model = load("acoustic", "3.0", "tf", precision="fp32", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_fp16() -> None:
  try:
    model = load("acoustic", "3.0", "tf", precision="fp16", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV3_0)


@pytest.mark.load_model
def test_v3_0_pt() -> None:
  ensure_torch_or_skip()

  try:
    model = load("acoustic", "3.0", "pt", precision="fp32")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV3_0)


@pytest.mark.load_model
def test_v3_0_onnx() -> None:
  ensure_onnxruntime_or_skip()

  try:
    model = load("acoustic", "3.0", "onnx", precision="fp32")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV3_0)


@pytest.mark.load_model
def test_v3_0_onnx_fp16() -> None:
  ensure_onnxruntime_or_skip()

  try:
    model = load("acoustic", "3.0", "onnx", precision="fp16")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV3_0)


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


def test_v3_0_pb_type_is_correct(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticPBDownloaderV3_0.get_model_path_and_labels",
    lambda lang: (Path("birdnet_v3_pb"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "pb")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticPBBackendFP32V3_0


def test_v3_0_tf_type_is_correct_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.tf_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticTFDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (Path("birdnet_v3.tflite"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "tf", precision="fp32")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticTFBackendFP32V3_0


def test_v3_0_tf_type_is_correct_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.tf_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticTFDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (Path("birdnet_v3.tflite"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "tf", precision="fp16")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticTFBackendFP16V3_0


def test_v3_0_onnx_type_is_correct_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticOnnxDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (Path("birdnet_v3.onnx"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "onnx", precision="fp32")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticOnnxBackendFP32V3_0


def test_v3_0_onnx_type_is_correct_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticOnnxDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (Path("birdnet_v3.onnx"), OrderedSet(["species_a"])),
  )

  model = load("acoustic", "3.0", "onnx", precision="fp16")
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is AcousticOnnxBackendFP16V3_0
