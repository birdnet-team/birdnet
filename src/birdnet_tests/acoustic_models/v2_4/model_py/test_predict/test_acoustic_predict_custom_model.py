import numpy.testing

from birdnet.model_loader import load_custom
from birdnet_tests.test_files import TEST_FILE_WAV, TEST_FILES_DIR


def test_custom_from_analyzer_v2_4_tf_fp32() -> None:
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier.tflite",
    TEST_FILES_DIR / "custom_models/tf/CustomClassifier_Labels.txt",
    library="tf",
    check_validity=False,
  )

  res = model.predict(TEST_FILE_WAV, top_k=None)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 4)
  numpy.testing.assert_almost_equal(mean, 0.1442, decimal=4)


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

  res = model.predict(TEST_FILE_WAV, top_k=None)
  mean = res.species_probs.mean()
  assert res.species_probs.shape == (1, 40, 4)
  numpy.testing.assert_almost_equal(mean, 0.1442, decimal=4)
