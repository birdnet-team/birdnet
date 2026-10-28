import tempfile
from pathlib import Path

import pytest

from birdnet.acoustic.models.v2_4.tf import (
  AcousticTFBackendFP32V2_4,
)
from birdnet.core.backends import TF_BACKEND_LIB_ARG, BackendLoader


def test_empty_file_can_not_be_loaded() -> None:
  with tempfile.NamedTemporaryFile(
    suffix=".tflite", delete=False, mode="wb"
  ) as empty_tflite:
    empty_tflite.write(b"")
  model_path = Path(empty_tflite.name)
  with pytest.raises(ValueError) as exc_info:  # noqa: F841
    BackendLoader.check_model_can_be_loaded(
      model_path,
      AcousticTFBackendFP32V2_4,
      {TF_BACKEND_LIB_ARG: "tflite"},
    )
  empty_tflite.close()
