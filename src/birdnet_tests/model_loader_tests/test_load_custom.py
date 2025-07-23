from typing import Literal, cast

from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.globals import MODEL_PRECISIONS
from birdnet.local_data import get_model_root_dir
from birdnet.model_loader import load_custom


def test_types_are_correct():
  assert (
    type(
      load_custom(
        "acoustic",
        "2.4",
        "pb",
        get_model_root_dir("acoustic", "2.4", "pb") / "model",
        get_model_root_dir("acoustic", "2.4", "pb") / "labels" / "en_us.txt",
      )
    )
    is AcousticPBModelV2_4
  )
  assert (
    type(
      load_custom(
        "acoustic",
        "2.4",
        "tf",
        get_model_root_dir("acoustic", "2.4", "tf") / "model-fp32.tflite",
        get_model_root_dir("acoustic", "2.4", "tf") / "labels" / "en_us.txt",
      )
    )
    is AcousticTFModelV2_4
  )
  assert (
    type(
      load_custom(
        "geo",
        "2.4",
        "pb",
        get_model_root_dir("geo", "2.4", "pb") / "model",
        get_model_root_dir("geo", "2.4", "pb") / "labels" / "en_us.txt",
      )
    )
    is GeoPBModelV2_4
  )
  assert (
    type(
      load_custom(
        "geo",
        "2.4",
        "tf",
        get_model_root_dir("geo", "2.4", "tf") / "model.tflite",
        get_model_root_dir("geo", "2.4", "tf") / "labels" / "en_us.txt",
      )
    )
    is GeoTFModelV2_4
  )


def test_types_with_precisions_are_correct():
  assert (
    type(
      load_custom(
        "acoustic",
        "2.4",
        "pb",
        get_model_root_dir("acoustic", "2.4", "pb") / "model",
        get_model_root_dir("acoustic", "2.4", "pb") / "labels" / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is AcousticPBModelV2_4
  )
  assert (
    type(
      load_custom(
        "acoustic",
        "2.4",
        "tf",
        get_model_root_dir("acoustic", "2.4", "tf") / "model-fp32.tflite",
        get_model_root_dir("acoustic", "2.4", "tf") / "labels" / "en_us.txt",
        precision=cast(MODEL_PRECISIONS, f"fp{32}"),
        check_validity=False,
      )
    )
    is AcousticTFModelV2_4
  )
  assert (
    type(
      load_custom(
        "geo",
        "2.4",
        "pb",
        get_model_root_dir("geo", "2.4", "pb") / "model",
        get_model_root_dir("geo", "2.4", "pb") / "labels" / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is GeoPBModelV2_4
  )
  assert (
    type(
      load_custom(
        "geo",
        "2.4",
        "tf",
        get_model_root_dir("geo", "2.4", "tf") / "model.tflite",
        get_model_root_dir("geo", "2.4", "tf") / "labels" / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is GeoTFModelV2_4
  )
