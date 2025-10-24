from typing import Literal, cast

import pytest

from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic_models.v2_4.pb import AcousticPBDownloaderV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.globals import MODEL_PRECISIONS
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.model_loader import load_custom


def test_v2_4_pb_with_library_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
    load_custom(
      "acoustic",
      "2.4",
      "pb",  # type: ignore
      get_model_path("acoustic", "2.4", "pb", "fp32"),
      get_lang_dir("acoustic", "2.4", "pb") / "en_us.txt",
      library="litert",
      check_validity=True,
    )  # type: ignore


def test_v2_4_pb() -> None:
  AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
  model = load_custom(
    "acoustic",
    "2.4",
    "pb",
    get_model_path("acoustic", "2.4", "pb", "fp32"),
    get_lang_dir("acoustic", "2.4", "pb") / "en_us.txt",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_tf_fp32() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp32"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tf",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_litert_fp32() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp32"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="litert",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_tf_fp16() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp16")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp16"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tf",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_litert_fp16() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp16")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp16"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="litert",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_tf_int8() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "int8")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "int8"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tf",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_v2_4_litert_int8() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "int8")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "int8"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="litert",
    check_validity=True,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_types_are_correct() -> None:
  AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")

  model_type, version, backend, precision = "acoustic", "2.4", "pb", "fp32"
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
    is AcousticModelV2_4
  )
  model_type, version, backend, precision = "acoustic", "2.4", "tf", "fp32"
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
    is AcousticModelV2_4
  )


def test_types_with_precisions_are_correct() -> None:
  AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")

  assert (
    type(
      load_custom(
        "acoustic",
        "2.4",
        "pb",
        get_model_path("acoustic", "2.4", "pb", "fp32"),
        get_lang_dir("acoustic", "2.4", "pb") / "en_us.txt",
        precision=cast(Literal["fp32"], f"fp{32}"),
        check_validity=False,
      )
    )
    is AcousticModelV2_4
  )
  assert (
    type(
      load_custom(
        "acoustic",
        "2.4",
        "tf",
        get_model_path("acoustic", "2.4", "tf", "fp32"),
        get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
        precision=cast(MODEL_PRECISIONS, f"fp{32}"),
        check_validity=False,
      )
    )
    is AcousticModelV2_4
  )
