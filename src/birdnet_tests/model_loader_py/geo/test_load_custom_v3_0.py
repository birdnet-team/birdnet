from typing import Literal, cast

import pytest

from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.model_loader import load_custom
from birdnet.utils.local_data import get_lang_dir, get_model_path
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.model_loader_py.acoustic.test_acoustic_load_custom import (
  check_validity,
)


def test_load_custom_geo_model_v3_0_tf_fp32() -> None:
  GeoTFDownloaderV3_0.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "geo",
    "3.0",
    "tf",
    get_model_path("geo", "3.0", "tf", "fp32"),
    get_lang_dir("geo", "3.0", "tf") / "en_us.txt",
    precision="fp32",
    library="tflite",
    check_validity=check_validity(),
  )
  assert isinstance(model, GeoModelV3_0)


def test_load_custom_geo_model_v3_0_tf_fp16() -> None:
  GeoTFDownloaderV3_0.get_model_path_and_labels("en_us", "fp16")
  model = load_custom(
    "geo",
    "3.0",
    "tf",
    get_model_path("geo", "3.0", "tf", "fp16"),
    get_lang_dir("geo", "3.0", "tf") / "en_us.txt",
    precision="fp16",
    library="tflite",
    check_validity=check_validity(),
  )
  assert isinstance(model, GeoModelV3_0)


def test_load_custom_geo_model_v3_0_tf_int8() -> None:
  GeoTFDownloaderV3_0.get_model_path_and_labels("en_us", "int8")
  model = load_custom(
    "geo",
    "3.0",
    "tf",
    get_model_path("geo", "3.0", "tf", "int8"),
    get_lang_dir("geo", "3.0", "tf") / "en_us.txt",
    precision="int8",
    library="tflite",
    check_validity=check_validity(),
  )
  assert isinstance(model, GeoModelV3_0)


@pytest.mark.litert
def test_load_custom_geo_model_v3_0_litert_fp32() -> None:
  ensure_litert_or_skip()
  GeoTFDownloaderV3_0.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "geo",
    "3.0",
    "tf",
    get_model_path("geo", "3.0", "tf", "fp32"),
    get_lang_dir("geo", "3.0", "tf") / "en_us.txt",
    precision="fp32",
    library="litert",
    check_validity=check_validity(),
  )
  assert isinstance(model, GeoModelV3_0)


def test_tf_types_are_correct() -> None:
  GeoTFDownloaderV3_0.get_model_path_and_labels("en_us", "fp32")
  model_type, version, backend, precision = "geo", "3.0", "tf", "fp32"
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
    is GeoModelV3_0
  )


def test_tf_type_with_precisions_is_correct() -> None:
  GeoTFDownloaderV3_0.get_model_path_and_labels("en_us", "fp32")
  assert (
    type(
      load_custom(
        "geo",
        "3.0",
        "tf",
        get_model_path("geo", "3.0", "tf", "fp32"),
        get_lang_dir("geo", "3.0", "tf") / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is GeoModelV3_0
  )
