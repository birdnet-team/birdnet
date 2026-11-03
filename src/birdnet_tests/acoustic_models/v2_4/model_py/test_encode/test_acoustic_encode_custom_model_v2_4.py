import pytest

from birdnet.model_loader import load_custom
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.test_files import (
  TEST_FILE_SHORT,
  TEST_FILE_SHORT_EMB_SHAPE,
  TEST_FILES_DIR,
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
    check_validity=False,
    is_raven=True,
  )

  with pytest.raises(  # noqa: SIM117
    ValueError,
    match=r"loaded backend does not support embeddings",
  ):
    with model.encode_session(n_workers=1):
      pass
