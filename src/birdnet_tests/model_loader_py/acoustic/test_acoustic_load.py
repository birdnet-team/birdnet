from typing import Literal, cast

import pytest
from requests.exceptions import ReadTimeout

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.globals import MODEL_PRECISIONS
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_litert_or_skip


@pytest.mark.litert
def test_pb_v2_4_with_library_raises_error() -> None:
  ensure_litert_or_skip()

  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    load("acoustic", "2.4", "pb", precision="fp32", library="litert")  # type: ignore


@pytest.mark.load_model
def test_v2_4_pb() -> None:
  try:
    model = load("acoustic", "2.4", "pb", precision="fp32")
  except ReadTimeout as e:
    # HTTPSConnectionPool(host='zenodo.org', port=443): Read timed out. (read timeout=30)
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.load_model
def test_v2_4_tf_fp32() -> None:
  try:
    model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.load_model
def test_v2_4_tf_fp16() -> None:
  try:
    model = load("acoustic", "2.4", "tf", precision="fp16", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_fp16() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.load_model
def test_v2_4_tf_int8() -> None:
  try:
    model = load("acoustic", "2.4", "tf", precision="int8", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_int8() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  assert isinstance(model, AcousticModelV2_4)


def test_pb_type_is_correct() -> None:
  assert type(load("acoustic", "2.4", "pb")) is AcousticModelV2_4


def test_tf_type_is_correct() -> None:
  assert type(load("acoustic", "2.4", "tf")) is AcousticModelV2_4


@pytest.mark.litert
def test_tf_litert_type_is_correct() -> None:
  ensure_litert_or_skip()

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
