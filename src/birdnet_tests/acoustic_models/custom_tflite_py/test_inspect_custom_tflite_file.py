from pathlib import Path

import pytest

from birdnet.acoustic.models import custom_tflite
from birdnet.acoustic.models.custom_tflite import (
  CustomTFLiteInfo,
  inspect_custom_tflite_file,
)
from birdnet_tests.acoustic_models.custom_tflite_py.stub_interpreter import (
  StubInterpreter,
  Tensor,
  v2_4_export,
  v3_0_export,
)

pytestmark = pytest.mark.no_tf


def _open_as(
  monkeypatch: pytest.MonkeyPatch, interp: StubInterpreter
) -> list[dict[str, object]]:
  """Make the inspection open `interp`; returns the calls of the loader."""
  opened: list[dict[str, object]] = []

  def fake_load_tf_model(path: Path, library: str, **kwargs: object) -> object:
    opened.append({"path": path, "library": library, **kwargs})
    return interp

  monkeypatch.setattr(custom_tflite, "load_tf_model", fake_load_tf_model)
  return opened


def test_the_tensors_of_the_opened_file_are_allocated(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The inspection doubles as the validity check: a file the interpreter
  cannot allocate must be rejected here, not by the first prediction."""
  interp = v2_4_export(3)
  opened = _open_as(monkeypatch, interp)

  info = inspect_custom_tflite_file(Path("classifier.tflite"), "litert")

  assert info == CustomTFLiteInfo("2.4", "replace", 3, False, True, 144_000)
  assert opened == [
    {"path": Path("classifier.tflite"), "library": "litert", "allocate_tensors": False}
  ]
  # A file that fixes its input length is allocated as it is.
  assert interp.calls == [("allocate",)]


def test_an_open_input_length_is_set_to_one_segment_before_allocating(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The official 3.0 export stores one sample for its open length, which
  LiteRT 2.2.0 and later refuse to allocate."""
  interp = v3_0_export(11_725)
  interp.inputs = [Tensor(0, "x", [1, -1])]
  _open_as(monkeypatch, interp)

  info = inspect_custom_tflite_file(Path("model.tflite"), "litert")

  assert info == CustomTFLiteInfo("3.0", "replace", 11_725, True, True, None)
  assert interp.calls == [("resize", 0, (1, 96_000)), ("allocate",)]
