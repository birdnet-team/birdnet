"""The pinned class count must follow the label list a model release ships."""

import pytest

from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
from birdnet_tests.helper import (
  ensure_not_intel_macos_or_skip,
  ensure_tf_2_20_or_skip,
)


@pytest.mark.load_model
def test_n_classes_equals_the_official_label_count() -> None:
  # The guards of load_perch_v2's tests: the download is only made where Perch runs.
  ensure_not_intel_macos_or_skip()
  ensure_tf_2_20_or_skip()

  model_dir, _ = AcousticPBDownloaderPerchV2.get_model_path_and_labels("CPU")
  labels_csv = (model_dir / "assets" / "labels.csv").read_text(encoding="utf-8")
  header, *labels = labels_csv.strip().splitlines()

  assert header == AcousticPBDownloaderPerchV2.LABELS_HEADER
  assert len(labels) == AcousticModelPerchV2.get_n_classes()
