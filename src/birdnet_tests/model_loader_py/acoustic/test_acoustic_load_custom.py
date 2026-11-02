"""`load_custom` on the official 2.4 SavedModel and the checked-in Raven export.

The TFLite paths are covered by `test_acoustic_load_custom_tflite_detection.py`
(stubbed), `test_official_models_through_load_custom.py` and `custom_exports/`
(real files).
"""

from pathlib import Path

import pytest

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.pb import (
  AcousticPBBackendFP32V2_4,
  AcousticPBDownloaderV2_4,
  AcousticRavenBackendFP32V2_4,
)
from birdnet.model_loader import load_custom
from birdnet.utils.local_data import get_lang_dir, get_model_path
from birdnet_tests.helper import ensure_validity_check_or_skip
from birdnet_tests.test_files import TEST_FILES_DIR

RAVEN_V2_4 = TEST_FILES_DIR / "custom_models" / "raven" / "CustomClassifier"
RAVEN_V2_4_LABELS = RAVEN_V2_4 / "labels" / "label_names.csv"


def test_v2_4_pb() -> None:
  # The only official SavedModel that goes through the validity check.
  ensure_validity_check_or_skip()
  AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
  model = load_custom(
    "acoustic",
    "2.4",
    "pb",
    get_model_path("acoustic", "2.4", "pb", "fp32"),
    get_lang_dir("acoustic", "2.4", "pb") / "en_us.txt",
    check_validity=True,
    precision="fp32",
    is_raven=False,
  )
  assert isinstance(model, AcousticModelV2_4)


def test_custom_from_analyzer_v2_4_as_no_raven_fp32_raises_exception() -> None:
  ensure_validity_check_or_skip()
  with pytest.raises(Exception, match=r"Failed to load model."):
    load_custom(
      "acoustic",
      "2.4",
      "pb",
      RAVEN_V2_4,
      RAVEN_V2_4_LABELS,
      check_validity=True,
      is_raven=False,
      precision="fp32",
    )


def test_v2_4_custom_pb_is_raven_by_default(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_path = tmp_path / "pb"
  variables = model_path / "variables"
  variables.mkdir(parents=True)
  (model_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")
  (variables / "variables.index").write_bytes(b"index")
  species_list = tmp_path / "species.txt"
  species_list.write_text("species_a\n", encoding="utf-8")
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded",
    lambda *args, **kwargs: 1,
  )

  default = load_custom("acoustic", "2.4", "pb", model_path, species_list)
  plain = load_custom("acoustic", "2.4", "pb", model_path, species_list, is_raven=False)
  assert default.backend_type is AcousticRavenBackendFP32V2_4
  assert plain.backend_type is AcousticPBBackendFP32V2_4
