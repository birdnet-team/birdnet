"""The pinned class count must follow the label list a model release ships."""

import pytest

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.utils.local_data import get_lang_dir

pytestmark = [pytest.mark.no_tf, pytest.mark.load_model]


def test_n_classes_equals_the_official_english_label_count() -> None:
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  lang_file = get_lang_dir("acoustic", "2.4", "tf") / "en_us.txt"
  labels = lang_file.read_text(encoding="utf-8").strip().splitlines()

  assert len(labels) == AcousticModelV2_4.get_n_classes()
