import numpy.testing
import pytest

from birdnet.acoustic_models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.model_loader import load_custom
from birdnet_tests.test_files import TEST_FILE_WAV, TEST_FILES_DIR


def test_custom_from_analyzer_v2_4_tf_fp32() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp32"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tf",
    precision="fp32",
    check_validity=False,
  )

  res = model.encode(TEST_FILE_WAV)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3406, rtol=1e-4)


def test_custom_from_analyzer_v2_4_tf_fp16() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp16")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "fp16"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tf",
    precision="fp16",
    check_validity=False,
  )

  res = model.encode(TEST_FILE_WAV)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3406, rtol=1e-4)


def test_custom_from_analyzer_v2_4_tf_int8() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "int8")
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    get_model_path("acoustic", "2.4", "tf", "int8"),
    get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt",
    library="tf",
    precision="int8",
    check_validity=False,
  )

  res = model.encode(TEST_FILE_WAV)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3347, rtol=1e-4)


def test_custom_from_analyzer_v2_4_litert_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier.tflite",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier_Labels.txt",
    library="litert",
    check_validity=False,
  )

  res = model.encode(TEST_FILE_WAV)
  mean = res.embeddings.mean()
  assert res.embeddings.shape == (1, 40, 1024)
  numpy.testing.assert_allclose(mean, 0.3406, rtol=1e-4)


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

  with pytest.raises(
    ValueError,
    match=r"loaded backend does not support embeddings",
  ):
    model.encode(TEST_FILE_WAV)
