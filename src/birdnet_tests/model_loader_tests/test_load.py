from typing import Literal, cast

from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.globals import MODEL_PRECISIONS
from birdnet.model_loader import load


def test_types_are_correct():
  assert type(load("acoustic", "2.4", "pb")) is AcousticPBModelV2_4
  assert type(load("acoustic", "2.4", "tf")) is AcousticTFModelV2_4
  assert type(load("geo", "2.4", "pb")) is GeoPBModelV2_4
  assert type(load("geo", "2.4", "tf")) is GeoTFModelV2_4


def test_types_with_precisions_are_correct():
  assert (
    type(load("acoustic", "2.4", "pb", precision=cast(Literal["fp32"], f"fp{32}")))
    is AcousticPBModelV2_4
  )
  assert (
    type(load("acoustic", "2.4", "tf", precision=cast(MODEL_PRECISIONS, f"fp{32}")))
    is AcousticTFModelV2_4
  )
  assert (
    type(load("geo", "2.4", "pb", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoPBModelV2_4
  )
  assert (
    type(load("geo", "2.4", "tf", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoTFModelV2_4
  )
