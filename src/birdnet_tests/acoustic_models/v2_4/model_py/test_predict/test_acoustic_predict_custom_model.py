import pytest

from birdnet.model_loader import load_custom
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.test_files import TEST_FILE_SHORT, TEST_FILES_DIR


def test_custom_from_analyzer_v2_4_tf_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier.tflite",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier_Labels.txt",
    library="tflite",
    check_validity=False,
  )

  res = model.predict(TEST_FILE_SHORT, top_k=None, n_workers=1)
  assert res.species_probs.shape == (1, 3, 4)


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

  res = model.predict(TEST_FILE_SHORT, top_k=None, n_workers=1)
  assert res.species_probs.shape == (1, 3, 4)
