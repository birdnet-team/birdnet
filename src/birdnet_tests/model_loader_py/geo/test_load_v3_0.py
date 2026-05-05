from typing import Literal, cast

import pytest
from requests import ReadTimeout

from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_litert_or_skip


@pytest.mark.litert
def test_pb_v3_0_with_library_raises_error() -> None:
  ensure_litert_or_skip()

  with pytest.raises(ValueError):
    load("geo", "3.0", "pb", precision="fp32")


@pytest.mark.load_model
def test_v3_0_tf_fp32() -> None:
  try:
    model = load("geo", "3.0", "tf", precision="fp32", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_fp16() -> None:
  try:
    model = load("geo", "3.0", "tf", precision="fp16", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.load_model
def test_v3_0_tf_int8() -> None:
  try:
    model = load("geo", "3.0", "tf", precision="int8", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.litert
def test_v3_0_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="fp32", library="litert")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.litert
def test_v3_0_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="fp16", library="litert")
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.litert
def test_v3_0_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("geo", "3.0", "tf", precision="int8", library="litert")
  assert isinstance(model, GeoModelV3_0)


def test_tf_tflite_fp32_type_is_correct() -> None:
  assert type(load("geo", "3.0", "tf", library="tflite")) is GeoModelV3_0


@pytest.mark.litert
def test_tf_litert_fp32_type_is_correct() -> None:
  ensure_litert_or_skip()

  assert type(load("geo", "3.0", "tf", library="litert")) is GeoModelV3_0


def test_types_with_precisions_are_correct() -> None:
  assert (
    type(load("geo", "3.0", "tf", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoModelV3_0
  )
  assert (
    type(load("geo", "3.0", "tf", precision=cast(Literal["fp16"], f"fp{16}")))
    is GeoModelV3_0
  )
  assert (
    type(load("geo", "3.0", "tf", precision=cast(Literal["int8"], f"int{8}")))
    is GeoModelV3_0
  )
