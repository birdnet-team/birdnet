from typing import Literal, cast

import pytest

from birdnet.geo_models.v2_4.model import GeoModelV2_4
from birdnet.geo_models.v2_4.pb import GeoPBDownloaderV2_4
from birdnet.geo_models.v2_4.tf import GeoTFDownloaderV2_4
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.model_loader import load_custom
from birdnet_tests.helper import ensure_litert_or_skip


def test_load_pb_with_custom_library_raises_error() -> None:
  ensure_litert_or_skip()

  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    GeoPBDownloaderV2_4.get_model_path_and_labels("en_us")
    load_custom(
      "geo",
      "2.4",
      "pb",  # type: ignore
      get_model_path("geo", "2.4", "pb", "fp32"),
      get_lang_dir("geo", "2.4", "pb") / "en_us.txt",
      library="litert",
      check_validity=True,
    )  # type: ignore


def test_load_custom_geo_model_v2_4_pb_fp32() -> None:
  GeoPBDownloaderV2_4.get_model_path_and_labels("en_us")
  model = load_custom(
    "geo",
    "2.4",
    "pb",
    get_model_path("geo", "2.4", "pb", "fp32"),
    get_lang_dir("geo", "2.4", "pb") / "en_us.txt",
    precision="fp32",
    check_validity=True,
  )
  assert isinstance(model, GeoModelV2_4)


def test_load_custom_geo_model_v2_4_tf_fp32() -> None:
  GeoTFDownloaderV2_4.get_model_path_and_labels("en_us")
  model = load_custom(
    "geo",
    "2.4",
    "tf",
    get_model_path("geo", "2.4", "tf", "fp32"),
    get_lang_dir("geo", "2.4", "tf") / "en_us.txt",
    library="tflite",
    check_validity=True,
  )
  assert isinstance(model, GeoModelV2_4)


@pytest.mark.litert
def test_load_custom_geo_model_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()
  GeoTFDownloaderV2_4.get_model_path_and_labels("en_us")
  model = load_custom(
    "geo",
    "2.4",
    "tf",
    get_model_path("geo", "2.4", "tf", "fp32"),
    get_lang_dir("geo", "2.4", "tf") / "en_us.txt",
    library="litert",
    check_validity=True,
  )
  assert isinstance(model, GeoModelV2_4)


def test_pb_types_are_correct() -> None:
  GeoPBDownloaderV2_4.get_model_path_and_labels("en_us")
  model_type, version, backend, precision = "geo", "2.4", "pb", "fp32"
  assert (
    type(
      load_custom(
        model_type,
        version,
        backend,
        get_model_path(model_type, version, backend, precision),
        get_lang_dir(model_type, version, backend) / "en_us.txt",
        check_validity=False,
      )
    )
    is GeoModelV2_4
  )


def test_tf_types_are_correct() -> None:
  GeoTFDownloaderV2_4.get_model_path_and_labels("en_us")
  model_type, version, backend, precision = "geo", "2.4", "tf", "fp32"
  assert (
    type(
      load_custom(
        model_type,
        version,
        backend,
        get_model_path(model_type, version, backend, precision),
        get_lang_dir(model_type, version, backend) / "en_us.txt",
        check_validity=False,
      )
    )
    is GeoModelV2_4
  )


def test_pb_type_with_precisions_is_correct() -> None:
  GeoPBDownloaderV2_4.get_model_path_and_labels("en_us")

  assert (
    type(
      load_custom(
        "geo",
        "2.4",
        "pb",
        get_model_path("geo", "2.4", "pb", "fp32"),
        get_lang_dir("geo", "2.4", "pb") / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is GeoModelV2_4
  )


def test_tf_type_with_precisions_is_correct() -> None:
  GeoTFDownloaderV2_4.get_model_path_and_labels("en_us")
  assert (
    type(
      load_custom(
        "geo",
        "2.4",
        "tf",
        get_model_path("geo", "2.4", "tf", "fp32"),
        get_lang_dir("geo", "2.4", "tf") / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is GeoModelV2_4
  )
