"""The pinned class count must follow the label list a model release ships."""

import pytest

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.tf import AcousticTFDownloaderV3_0

pytestmark = [pytest.mark.no_tf, pytest.mark.load_model]


def test_n_classes_equals_the_official_english_label_count() -> None:
  AcousticTFDownloaderV3_0.ensure_labels_available()
  lang_file = AcousticTFDownloaderV3_0.get_lang_file("en_us")
  labels = lang_file.read_text(encoding="utf-8").strip().splitlines()

  assert len(labels) == AcousticModelV3_0.get_n_classes()
