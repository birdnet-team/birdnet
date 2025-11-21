import pytest

from birdnet.acoustic_models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.model_loader import load_custom
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.test_files import (
  TEST_FILE_SHORT,
  TEST_FILE_SHORT_EMB_SHAPE,
  TEST_FILES_DIR,
)


def test_custom_from_analyzer_v2_4_tf_fp32() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp32"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tflite",
    precision="fp32",
    check_validity=False,
  )

  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_custom_from_analyzer_v2_4_tf_fp16() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp16")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp16"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tflite",
    precision="fp16",
    check_validity=False,
  )

  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_custom_from_analyzer_v2_4_tf_int8() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "int8")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "int8"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tflite",
    precision="int8",
    check_validity=False,
  )

  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


@pytest.mark.litert
def test_custom_from_analyzer_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier.tflite",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier_Labels.txt",
    library="litert",
    check_validity=False,
  )

  with model.encode_session(n_workers=1) as session:
    res = session.run(TEST_FILE_SHORT)
  assert res.embeddings.shape == TEST_FILE_SHORT_EMB_SHAPE


def test_custom_from_analyzer_v2_4_raven_fp32_raise_exception() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "pb",
    TEST_FILES_DIR / "custom_models/raven/CustomClassifier",
    TEST_FILES_DIR / "custom_models/raven/CustomClassifier/labels/label_names.csv",
    check_validity=True,
    is_raven=True,
  )

  with pytest.raises(  # noqa: SIM117
    ValueError,
    match=r"loaded backend does not support embeddings",
  ):
    with model.encode_session(n_workers=1):
      pass
