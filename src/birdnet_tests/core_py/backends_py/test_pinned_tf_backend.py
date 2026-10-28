"""How a TFLite backend gets its tensor indices.

An official model pins them at class level; a custom classifier resolves them
from the file, so it has no class-level indices at all.
"""

from pathlib import Path

import pytest

from birdnet.acoustic.models.custom_tflite import CustomTFLiteBackend
from birdnet.core.backends import PinnedTFBackend, TFBackend
from birdnet.globals import MODEL_PRECISIONS

pytestmark = pytest.mark.no_tf


def _construct(backend_type: type[TFBackend]) -> TFBackend:
  return backend_type(Path("model.file"), "CPU", False, inference_library="tflite")


class _Pinned(PinnedTFBackend):
  @classmethod
  def in_idx(cls) -> int:
    return 3

  @classmethod
  def prediction_out_idx(cls) -> int:
    return 7

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp32"

  @classmethod
  def supports_encoding(cls) -> bool:
    return False


class _Unresolved(TFBackend):
  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp32"

  @classmethod
  def supports_encoding(cls) -> bool:
    return False


def test_a_pinned_backend_resolves_its_class_level_indices() -> None:
  backend = _construct(_Pinned)

  assert backend._resolve_tensor_indices(object()) == (3, 7, None)


def test_a_backend_that_cannot_resolve_its_indices_is_abstract() -> None:
  with pytest.raises(TypeError, match="_resolve_tensor_indices"):
    _construct(_Unresolved)  # type: ignore[type-abstract]


@pytest.mark.parametrize("hook", ["in_idx", "prediction_out_idx", "encoding_out_idx"])
def test_a_custom_classifier_pins_no_indices(hook: str) -> None:
  assert not hasattr(CustomTFLiteBackend, hook)
