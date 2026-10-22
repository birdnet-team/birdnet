from pathlib import Path

import pytest

from birdnet.acoustic.models import custom_tflite
from birdnet.acoustic.models.custom_tflite import (
  CustomTFLiteInfo,
  inspect_custom_tflite_file,
)
from birdnet_tests.acoustic_models.custom_tflite_py.stub_interpreter import (
  v2_4_export,
)

pytestmark = pytest.mark.no_tf


def test_the_file_is_opened_with_its_tensors_allocated(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The inspection doubles as the validity check: a file the interpreter
  cannot allocate must be rejected here, not by the first prediction."""
  opened: list[dict[str, object]] = []

  def fake_load_tf_model(path: Path, library: str, **kwargs: object) -> object:
    opened.append({"path": path, "library": library, **kwargs})
    return v2_4_export(3)

  monkeypatch.setattr(custom_tflite, "load_tf_model", fake_load_tf_model)

  info = inspect_custom_tflite_file(Path("classifier.tflite"), "litert")

  assert info == CustomTFLiteInfo("2.4", "replace", 3, False, True, 144_000)
  assert opened == [
    {"path": Path("classifier.tflite"), "library": "litert", "allocate_tensors": True}
  ]
