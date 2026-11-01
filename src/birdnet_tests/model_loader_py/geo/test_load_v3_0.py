import pytest

from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_onnxruntime_or_skip,
  ensure_tf_2_19_or_2_18,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)


@pytest.mark.load_model
def test_v3_0_pb() -> None:
  model = load("geo", "3.0", "pb", precision="fp32")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.no_tf
@pytest.mark.load_model
def test_v3_0_pt() -> None:
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()
  model = load("geo", "3.0", "pt", precision="fp32")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.no_tf
@pytest.mark.load_model
def test_v3_0_onnx_fp32() -> None:
  ensure_onnxruntime_or_skip()
  model = load("geo", "3.0", "onnx", precision="fp32")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.no_tf
@pytest.mark.load_model
def test_v3_0_onnx_fp16() -> None:
  ensure_onnxruntime_or_skip()
  model = load("geo", "3.0", "onnx", precision="fp16")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_fp32() -> None:
  ensure_tf_2_19_or_2_18()
  model = load("geo", "3.0", "tf", precision="fp32", library="tflite")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_fp16() -> None:
  ensure_tf_2_19_or_2_18()
  model = load("geo", "3.0", "tf", precision="fp16", library="tflite")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_int8() -> None:
  ensure_tf_2_19_or_2_18()
  model = load("geo", "3.0", "tf", precision="int8", library="tflite")
  assert isinstance(model, GeoModelV3_0)


def test_v3_0_litert_raises_error() -> None:
  with pytest.raises(
    RuntimeError,
    match=r"geo model v3\.0 TF backend is not supported with ai_edge_litert",
  ):
    load("geo", "3.0", "tf", precision="fp32", library="litert")


def test_pb_type_is_correct() -> None:
  assert type(load("geo", "3.0", "pb")) is GeoModelV3_0
