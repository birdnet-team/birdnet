from typing import Literal, cast

import pytest

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
from birdnet.acoustic.models.v2_4.tf import (
  AcousticTFBackendFP32CustomAppendHiddenV2_4,
  AcousticTFBackendFP32CustomAppendV2_4,
  AcousticTFBackendFP32CustomReplaceHiddenV2_4,
  AcousticTFBackendFP32V2_4,
  AcousticTFDownloaderV2_4,
)
from birdnet.globals import MODEL_PRECISIONS
from birdnet.model_loader import load_custom
from birdnet.utils.helper import check_is_intel_macos, check_is_python_312
from birdnet.utils.local_data import get_lang_dir, get_model_path
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILES_DIR


def check_validity() -> bool:
  # Note: on Intel macOS python 3.12 the model could not be loaded for unknown reasons
  # it will result in a TimeoutError
  check_it = not (check_is_python_312() and check_is_intel_macos())
  return check_it


@pytest.mark.litert
def test_v2_4_pb_with_library_raises_error() -> None:
  ensure_litert_or_skip()

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
      precision="fp32",
      check_validity=check_validity(),
      is_raven=False,
    )  # type: ignore


def test_v2_4_pb() -> None:
  AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
  model = load_custom(
    "acoustic",
    "2.4",
    "pb",
    get_model_path("acoustic", "2.4", "pb", "fp32"),
    get_lang_dir("acoustic", "2.4", "pb") / "en_us.txt",
    check_validity=check_validity(),
    precision="fp32",
    is_raven=False,
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
    library="tflite",
    precision="fp32",
    check_validity=check_validity(),
  )
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()

  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp32"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="litert",
    precision="fp32",
    check_validity=check_validity(),
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
    library="tflite",
    precision="fp16",
    check_validity=check_validity(),
  )
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_fp16() -> None:
  ensure_litert_or_skip()

  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp16")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp16"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="litert",
    precision="fp16",
    check_validity=check_validity(),
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
    library="tflite",
    precision="int8",
    check_validity=check_validity(),
  )
  assert isinstance(model, AcousticModelV2_4)


@pytest.mark.litert
def test_v2_4_litert_int8() -> None:
  ensure_litert_or_skip()

  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "int8")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "int8"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="litert",
    precision="int8",
    check_validity=check_validity(),
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
        precision=precision,
        is_raven=False,
      )
    )
    is AcousticModelV2_4
  )

  assert (
    type(
      load_custom(
        model_type,
        version,
        backend,
        get_model_path(model_type, version, backend, precision),
        get_lang_dir(model_type, version, backend) / "en_us.txt",
        check_validity=False,
        is_raven=True,
        precision=precision,
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
        precision=precision,
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
        is_raven=False,
      )
    )
    is AcousticModelV2_4
  )

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
        is_raven=True,
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


@pytest.mark.litert
def test_custom_from_analyzer_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/replace.tflite",
    TEST_FILES_DIR / "custom_models/tf/replace_Labels.txt",
    library="litert",
    check_validity=True,
    precision="fp32",
    classifier_type="replace",
  )
  assert isinstance(model, AcousticModelV2_4)


def test_custom_from_analyzer_v2_4_tf_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/replace.tflite",
    TEST_FILES_DIR / "custom_models/tf/replace_Labels.txt",
    library="tflite",
    check_validity=check_validity(),
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)


def test_custom_from_analyzer_v2_4_raven_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "pb",
    TEST_FILES_DIR / "custom_models/raven/CustomClassifier",
    TEST_FILES_DIR / "custom_models/raven/CustomClassifier/labels/label_names.csv",
    check_validity=check_validity(),
    is_raven=True,
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)


def test_custom_from_analyzer_v2_4_as_no_raven_fp32_raises_exception() -> None:
  if not check_validity():
    pytest.skip(
      "Skipping test on Intel macOS Python 3.12 where model loading fails unexpectedly."
    )

  with pytest.raises(
    Exception,
    match=r"Failed to load model.",
  ):
    model = load_custom(
      "acoustic",
      "2.4",
      "pb",
      TEST_FILES_DIR / "custom_models/raven/CustomClassifier",
      TEST_FILES_DIR / "custom_models/raven/CustomClassifier/labels/label_names.csv",
      check_validity=True,
      is_raven=False,
      precision="fp32",
    )
    assert isinstance(model, AcousticModelV2_4)


def test_custom_from_analyzer_v2_4_append_tf_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/append.tflite",
    TEST_FILES_DIR / "custom_models/tf/append_Labels.txt",
    library="tflite",
    check_validity=check_validity(),
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)
  if check_validity():
    assert model.backend_type is AcousticTFBackendFP32CustomAppendV2_4


@pytest.mark.litert
def test_custom_from_analyzer_v2_4_append_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/append.tflite",
    TEST_FILES_DIR / "custom_models/tf/append_Labels.txt",
    library="litert",
    check_validity=True,
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)
  assert model.backend_type is AcousticTFBackendFP32CustomAppendV2_4


def test_custom_from_analyzer_v2_4_replace_hidden_tf_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/replace+hidden1024.tflite",
    TEST_FILES_DIR / "custom_models/tf/replace+hidden1024_Labels.txt",
    library="tflite",
    check_validity=check_validity(),
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)
  if check_validity():
    assert model.backend_type is AcousticTFBackendFP32CustomReplaceHiddenV2_4


@pytest.mark.litert
def test_custom_from_analyzer_v2_4_replace_hidden_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/replace+hidden1024.tflite",
    TEST_FILES_DIR / "custom_models/tf/replace+hidden1024_Labels.txt",
    library="litert",
    check_validity=True,
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)
  assert model.backend_type is AcousticTFBackendFP32CustomReplaceHiddenV2_4


def test_custom_from_analyzer_v2_4_append_hidden_tf_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/append+hidden1024.tflite",
    TEST_FILES_DIR / "custom_models/tf/append+hidden1024_Labels.txt",
    library="tflite",
    check_validity=check_validity(),
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)
  if check_validity():
    assert model.backend_type is AcousticTFBackendFP32CustomAppendHiddenV2_4


@pytest.mark.litert
def test_custom_from_analyzer_v2_4_append_hidden_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/append+hidden1024.tflite",
    TEST_FILES_DIR / "custom_models/tf/append+hidden1024_Labels.txt",
    library="litert",
    check_validity=True,
    precision="fp32",
  )
  assert isinstance(model, AcousticModelV2_4)
  assert model.backend_type is AcousticTFBackendFP32CustomAppendHiddenV2_4


def test_custom_classifier_types_explicit_no_check_validity() -> None:
  cases = [
    (
      "custom_models/tf/append.tflite",
      "custom_models/tf/append_Labels.txt",
      "append",
      AcousticTFBackendFP32CustomAppendV2_4,
    ),
    (
      "custom_models/tf/replace+hidden1024.tflite",
      "custom_models/tf/replace+hidden1024_Labels.txt",
      "replace_hidden",
      AcousticTFBackendFP32CustomReplaceHiddenV2_4,
    ),
    (
      "custom_models/tf/append+hidden1024.tflite",
      "custom_models/tf/append+hidden1024_Labels.txt",
      "append_hidden",
      AcousticTFBackendFP32CustomAppendHiddenV2_4,
    ),
  ]
  for model_file, labels_file, classifier_type, expected_backend in cases:
    model = load_custom(
      "acoustic",
      "2.4",
      "tf",
      TEST_FILES_DIR / model_file,
      TEST_FILES_DIR / labels_file,
      library="tflite",
      check_validity=False,
      precision="fp32",
      classifier_type=classifier_type,
    )
    assert isinstance(model, AcousticModelV2_4)
    assert model.backend_type is expected_backend


def test_custom_replace_explicit_no_check_validity() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/replace.tflite",
    TEST_FILES_DIR / "custom_models/tf/replace_Labels.txt",
    library="tflite",
    check_validity=False,
    precision="fp32",
    classifier_type="replace",
  )
  assert isinstance(model, AcousticModelV2_4)
  assert model.backend_type is AcousticTFBackendFP32V2_4


def test_custom_species_mismatch_raises_error() -> None:
  if not check_validity():
    pytest.skip(
      "Skipping test on Intel macOS Python 3.12 where model loading fails unexpectedly."
    )

  with pytest.raises(
    ValueError,
    match=r"species",
  ):
    load_custom(
      "acoustic",
      "2.4",
      "tf",
      TEST_FILES_DIR / "custom_models/tf/append.tflite",
      # intentionally wrong labels file (replace has a different species count)
      TEST_FILES_DIR / "custom_models/tf/replace_Labels.txt",
      library="tflite",
      check_validity=True,
      precision="fp32",
    )
