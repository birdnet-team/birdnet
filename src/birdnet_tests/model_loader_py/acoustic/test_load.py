from typing import Literal, cast

import pytest

from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.globals import MODEL_PRECISIONS
from birdnet.model_loader import load


def test_pb_v2_4_with_library_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    load("acoustic", "2.4", "pb", precision="fp32", library="litert")  # type: ignore


def test_v2_4_pb() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_tf_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_litert_fp32() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_tf_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_litert_fp16() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_tf_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_litert_int8() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  assert isinstance(model, AcousticModelV2_4)


def test_types_are_correct() -> None:
  assert type(load("acoustic", "2.4", "pb")) is AcousticModelV2_4
  assert type(load("acoustic", "2.4", "pb")) is AcousticModelV2_4
  assert type(load("acoustic", "2.4", "tf")) is AcousticModelV2_4
  assert type(load("acoustic", "2.4", "tf", library="litert")) is AcousticModelV2_4


def test_types_with_precisions_are_correct() -> None:
  assert (
    type(load("acoustic", "2.4", "pb", precision=cast(Literal["fp32"], f"fp{32}")))
    is AcousticModelV2_4
  )
  assert (
    type(load("acoustic", "2.4", "tf", precision=cast(MODEL_PRECISIONS, f"fp{32}")))
    is AcousticModelV2_4
  )
