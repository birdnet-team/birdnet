from typing import Literal, cast

import pytest

from birdnet.geo_models.v2_4.model import GeoModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_litert_or_skip


@pytest.mark.litert
def test_pb_v2_4_with_library_raises_error() -> None:
  ensure_litert_or_skip()

  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    load("geo", "2.4", "pb", precision="fp32", library="litert")  # type: ignore


@pytest.mark.load_model
def test_v2_4_pb() -> None:
  model = load("geo", "2.4", "pb", precision="fp32")
  assert isinstance(model, GeoModelV2_4)


@pytest.mark.load_model
def test_v2_4_tf_fp32() -> None:
  model = load("geo", "2.4", "tf", precision="fp32", library="tflite")
  assert isinstance(model, GeoModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  assert isinstance(model, GeoModelV2_4)


def test_pb_type_is_correct() -> None:
  assert type(load("geo", "2.4", "pb")) is GeoModelV2_4


def test_tf_tflite_type_is_correct() -> None:
  assert type(load("geo", "2.4", "tf", library="tflite")) is GeoModelV2_4


@pytest.mark.litert
def test_tf_litert_type_is_correct() -> None:
  ensure_litert_or_skip()

  assert type(load("geo", "2.4", "tf", library="litert")) is GeoModelV2_4


def test_types_with_precisions_are_correct() -> None:
  assert (
    type(load("geo", "2.4", "pb", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoModelV2_4
  )
  assert (
    type(load("geo", "2.4", "tf", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoModelV2_4
  )
